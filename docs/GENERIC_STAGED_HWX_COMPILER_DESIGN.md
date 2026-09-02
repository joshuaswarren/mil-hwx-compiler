# Generic Staged MIL-to-HWX Compiler

Status: approved implementation design  
Target: Apple M4 H16G  
Scope: research compiler for the decoded HWX primitive surface

## Goal

Compile any legal graph composed from decoded ANE primitives into a new HWX
object. The compiler must derive fusion, tiles, surfaces, DMA, task order and
descriptor fields in stages. It must not select a workload by name, copy a
recovered task-descriptor row, or patch a compiler-minted HWX skeleton.

“Any graph” is deliberately bounded by current hardware knowledge. An operation
or shape whose H16G descriptor grammar is not decoded fails with a precise
diagnostic. Adding support means adding a primitive capability or target-fact
row, not adding a whole-graph recognizer.

## Non-goals

- Reproduce every optimization in Apple’s production compiler.
- Accept every Core ML or MIL operation.
- Reproduce Apple’s private IR classes or exact pass names.
- Find globally optimal schedules.
- Replace `aned` program lifecycle on stock macOS.

## Pipeline

```text
MIL text and blobs
  -> typed operation graph
  -> normalize and decompose
  -> form legal fused regions
  -> H16G legalize
  -> tile, allocate and schedule
  -> scheduled HWX graph
  -> command and TD encoding
  -> HWX object construction
  -> structural verification
```

There are two compiler IRs.

### Operation graph

The existing typed SSA graph becomes the compiler’s operation graph. Operations
remain individual nodes. A fused region is a list of nodes plus internal edges;
it is not a new `Attention` or `W8A8Chain` operation.

Each node carries:

- operation kind;
- typed input and output values;
- tensor shape and layout;
- scalar attributes;
- constant reference when applicable;
- quantization state;
- source location.

The initial primitive kinds are Conv, Matmul, ALU, LUT, Reduce and Layout. DMA
and Wait are scheduling operations and do not appear in the operation graph.

### Scheduled HWX graph

The scheduled graph contains only decisions needed to emit and execute HWX:

- logical and physical surfaces;
- tile ranges;
- SRAM allocations and lifetimes;
- constant sections;
- DMA Load, Inter and Store commands;
- compute commands;
- task dependencies and waits;
- numeric mode;
- target-selected descriptor operands.

It contains no opaque descriptor row. Unknown fixed target values may exist as
named H16G facts with provenance, but every output word is written by a field
encoder.

## Passes

### 1. Normalize

- resolve and verify SSA;
- materialize BLOBFILE metadata;
- fold scalar constants;
- remove dead operations;
- eliminate common subexpressions;
- collapse identity reshape and inverse transpose pairs;
- canonicalize convolution, axes and quantization attributes.

### 2. Decompose

Lower compound operations to decoded primitives. The first decompositions are:

```text
softmax -> reduce_max, sub, LUT(exp), reduce_sum, reciprocal, mul
layer_norm -> reduce_mean, sub, square, reduce_mean, rsqrt, mul, add
```

Simple Conv, Matmul, ALU, LUT, Reduce and Layout operations pass through.

### 3. Fuse

Build producer-consumer regions without pattern names. Two adjacent operations
may share a region when:

- the edge is dataflow-compatible;
- no external use requires an intermediate materialization;
- both operations have a legal H16G implementation;
- layouts can be reconciled;
- the estimated live working set fits the target budget;
- fusion does not introduce an unsupported numeric transition.

The first selector is deterministic greedy region growth in graph order. It is
not intended to be globally optimal. Fusion runs again after legalization so a
layout or quantization decision can expose or break a region.

### 4. Legalize

- assign physical layouts;
- validate primitive capability and shape constraints;
- propagate quantization state;
- select W8A8 boundary, packed and output modes per Conv node;
- insert required layout and dtype conversions;
- split operations only where a decoded tiling law exists.

Target decisions read `H16GTargetFacts`. Passes contain no H16G numeric
literals.

### 5. Plan

For each region:

1. choose tiles using decoded target laws;
2. compute value liveness;
3. allocate SRAM with deterministic first-fit reuse;
4. assign 16-byte-granule addresses across 64 banks;
5. classify constants as resident or streamed;
6. insert DMA Load for external inputs;
7. use DMA Inter for internal region edges;
8. insert one final DMA Store per materialized result;
9. create task dependencies and conservative hazard waits.

The first planner is intentionally simple. It must be correct and inspectable;
later cost models can replace its tile choice without changing either IR.

### 6. Emit

Each primitive has one H16G command encoder. Encoders consume scheduled command
objects and target facts. They never inspect a workload name or the surrounding
graph.

The TD writer starts with a zeroed byte buffer and writes named fields. The HWX
object writer builds headers, load commands, sections, symbols, alignments and
contents from data structures. It never loads and modifies an existing HWX.

## How the three evidence workloads compile

### Conv and ReLU

Two ordinary graph nodes form one region. Legalization selects a Conv command
with an ALU epilogue. Planning emits input/weight loads, compute and output
store. The generic Conv encoder emits the TD.

### Attention

Transpose, Matmul, ALU, Reduce and LUT nodes form one legal region when its live
set fits. Planning keeps intermediate values in SRAM and emits a multi-TD chain.
No Attention node or emitter exists.

### W8A8 convolution chain

Quantization propagation marks each ordinary Conv node as input boundary,
packed W8A8 or output boundary. Planning gives internal activations DMA Inter
edges. The same Conv encoder writes mode, dtype, scale and DMA fields from each
node’s numeric state.

## Extension contract

A new operation requires:

1. an operation kind or decomposition into existing kinds;
2. shape and dtype verification;
3. a capability row;
4. one command encoder if no existing primitive represents it;
5. at least two structurally different tests;
6. M4 numerical verification before it is marked supported.

It does not require a new graph recognizer or workload emitter.

## Verification

- Pass dumps show the graph before and after every transformation.
- Every pass has negative and positive unit tests.
- Equivalent graph spellings normalize to the same operation graph.
- Fusion tests include external-use, layout, live-set and quantization barriers.
- Planner tests check lifetime reuse, alignment, bank mapping and DMA roles.
- TD tests compare named decoded fields, not whole copied descriptor rows.
- Object tests build from an empty buffer and reparse the result.
- Conv+ReLU, attention and W8A8 execute twice on M4 and compare element-wise
  with independent CPU references.
- At least two legal variants per primitive path are required before the path is
  described as generic.

## Migration

The v1 implementation remains unchanged as an oracle until replacement gates
pass. New code lives under `lib/Transform`, `lib/Planning` and
`plugins/H16G/Encoding`. Production registration moves to the staged pipeline
one workload at a time. Legacy files remain on disk but are excluded from the
production build after all three gates pass.
