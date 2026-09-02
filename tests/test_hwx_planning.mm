#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEDecomposePass.h"
#import "ANEFusionPass.h"
#import "ANEGraphVerifier.h"
#import "ANEH16GLegalizePass.h"
#import "ANEScheduledGraph.h"
#import "ANEMemoryPlanner.h"
#import "ANETaskScheduler.h"
#import "H16GTarget.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"

#include <stdio.h>

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

static ANEOperationGraph *fixture(NSString *name,
                                  ANEDiagnosticEngine *diagnostics) {
    NSString *path = [@"tests/fixtures" stringByAppendingPathComponent:name];
    MILLexer *lexer = [[MILLexer alloc]
        initWithData:[NSData dataWithContentsOfFile:path] diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
        diagnostics:diagnostics];
    MILProgramSyntax *syntax = [parser parseProgram];
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics])
        return nil;
    ANEOperationGraph *graph = [[ANEOperationGraph alloc]
        initWithFunction:module.functions[0] diagnostics:diagnostics];
    H16GTarget *target = [H16GTarget currentTarget];
    if (![ANEDecomposePass runOnGraph:graph diagnostics:diagnostics] ||
        ![ANEFusionPass runOnGraph:graph target:target diagnostics:diagnostics] ||
        ![ANEH16GLegalizePass runOnGraph:graph target:target diagnostics:diagnostics])
        return nil;
    return graph;
}

static ANEOperationGraph *graphFromSource(NSString *source,
                                          ANEDiagnosticEngine *diagnostics) {
    MILLexer *lexer = [[MILLexer alloc]
        initWithData:[source dataUsingEncoding:NSUTF8StringEncoding]
        diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
        diagnostics:diagnostics];
    MILProgramSyntax *syntax = [parser parseProgram];
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics])
        return nil;
    ANEOperationGraph *graph = [[ANEOperationGraph alloc]
        initWithFunction:module.functions[0] diagnostics:diagnostics];
    H16GTarget *target = [H16GTarget currentTarget];
    if (![ANEDecomposePass runOnGraph:graph diagnostics:diagnostics] ||
        ![ANEFusionPass runOnGraph:graph target:target diagnostics:diagnostics] ||
        ![ANEH16GLegalizePass runOnGraph:graph target:target diagnostics:diagnostics])
        return nil;
    return graph;
}

static NSUInteger commandCount(ANEScheduledGraph *graph,
                               ANEScheduledCommandKind kind) {
    NSUInteger count = 0;
    for (ANEScheduledTask *task in graph.tasks)
        for (ANEScheduledCommand *command in task.commands)
            if (command.kind == kind) count++;
    return count;
}

static ANEScheduledSurface *surfaceNamed(ANEScheduledGraph *graph,
                                         NSString *identifier) {
    for (ANEScheduledSurface *surface in graph.surfaces)
        if ([surface.identifier isEqualToString:identifier]) return surface;
    return nil;
}

static void dumpTaskGroups(ANEScheduledGraph *graph) {
    fprintf(stderr, "actual scheduled task groups (%lu):\n",
            (unsigned long)graph.tasks.count);
    for (ANEScheduledTask *task in graph.tasks) {
        fprintf(stderr, "  task %lu: %s\n", (unsigned long)task.index,
                [[task.sourceNodeIdentifiers componentsJoinedByString:@", "]
                    UTF8String]);
    }
}

static void testConvReluBecomesOneScheduledComputeTask(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = fixture(@"conv_relu.mil", diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    expect(scheduled != nil && diagnostics.errorCount == 0,
           @"Conv and ReLU schedule successfully");
    expect(scheduled.tasks.count == 1,
           @"a legal elementwise epilogue does not become a second task");
    expect(scheduled.tasks.firstObject.operationKind == ANEOperationKindConv,
           @"the fused task retains the Conv command kind");
    expect([scheduled.tasks.firstObject.nodeIdentifier isEqualToString:@"y"],
           @"the fused task produces the epilogue result");
    expect(commandCount(scheduled, ANEScheduledCommandKindCompute) == 1 &&
           commandCount(scheduled, ANEScheduledCommandKindDMAStore) == 1,
           @"the fused task has one compute and one final store");
    ANEScheduledSurface *output = surfaceNamed(scheduled,@"y");
    expect(output.role == ANEScheduledSurfaceRoleOutput &&
           !output.sramAllocated && scheduled.peakSRAMBytes == 0,
           @"external output IOSurfaces are not counted as resident SRAM");
}

static void testAttentionUsesInterTaskDMA(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = fixture(@"attention.mil", diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    expect(scheduled != nil && diagnostics.errorCount == 0,
           @"decomposed attention schedules");
    if (scheduled.tasks.count != 10) {
        dumpTaskGroups(scheduled);
        fprintf(stderr, "%s", operations.textualDescription.UTF8String);
    }
    expect(scheduled.tasks.count == 10,
           @"structural task fusion maps seventeen graph ops to ten hardware tasks");
    NSArray<NSArray<NSString *> *> *expectedGroups = @[
        @[@"Q0", @"K0", @"V0"],
        @[@"Q", @"K"],
        @[@"scores"],
        @[@"scaled"],
        @[@"probabilities.max"],
        @[@"probabilities.centered", @"probabilities.exp"],
        @[@"probabilities.sum"],
        @[@"probabilities.reciprocal"],
        @[@"probabilities", @"V"],
        @[@"context", @"context_nchw", @"y"],
    ];
    for (NSUInteger i = 0; i < expectedGroups.count &&
                           i < scheduled.tasks.count; ++i)
        expect([scheduled.tasks[i].sourceNodeIdentifiers
                isEqualToArray:expectedGroups[i]],
               @"hardware tasks retain their structurally fused source nodes");
    expect(commandCount(scheduled, ANEScheduledCommandKindDMAInter) >= 8,
           @"fused primitive edges remain in SRAM through DMA_INTER");
    expect(commandCount(scheduled, ANEScheduledCommandKindDMAStore) == 1,
           @"only the graph result is stored to DRAM");
    ANEScheduledSurface *input = surfaceNamed(scheduled, @"x");
    expect(input.elementType == ANEElementTypeFP16 &&
           [input.shape isEqualToArray:@[@1,@64,@4,@192]],
           @"external surface uses the declared function-input type");
    for (ANEScheduledTask *task in scheduled.tasks)
        expect(![task.nodeIdentifier containsString:@"attention"],
               @"scheduler never receives a workload-named node");
}

static void testW8A8PlannerKeepsFourConvModes(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = fixture(@"w8a8_conv_chain.mil", diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    NSMutableArray<ANEScheduledTask *> *convTasks = [NSMutableArray array];
    for (ANEScheduledTask *task in scheduled.tasks)
        if (task.operationKind == ANEOperationKindConv) [convTasks addObject:task];
    expect(convTasks.count == 4, @"folded Q/DQ nodes create no fake compute tasks");
    NSArray<NSNumber *> *modes = @[
        @(ANELegalNumericModeW8A8InputBoundary),
        @(ANELegalNumericModeW8A8Packed),
        @(ANELegalNumericModeW8A8Packed),
        @(ANELegalNumericModeW8A8OutputBoundary),
    ];
    for (NSUInteger i = 0; i < convTasks.count; ++i)
        expect(convTasks[i].numericMode == modes[i].unsignedIntegerValue,
               @"scheduled Conv retains legalized numeric mode");
    expect(commandCount(scheduled, ANEScheduledCommandKindDMAInter) == 3,
           @"three quantized activation bridges stay internal");
    expect(commandCount(scheduled, ANEScheduledCommandKindDMAStore) == 1,
           @"the four-layer chain has one final store");
}

static void testSRAMAllocationsAreAlignedAndReused(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = fixture(@"w8a8_conv_chain.mil", diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    NSMutableArray<ANEScheduledSurface *> *activations = [NSMutableArray array];
    for (ANEScheduledSurface *surface in scheduled.surfaces) {
        if (!surface.sramAllocated) continue;
        expect(surface.sramOffset % 16 == 0, @"SRAM allocations use 16-byte granules");
        expect(surface.bank == (surface.sramOffset / 16) % 64,
               @"bank mapping follows the recovered interleave");
        [activations addObject:surface];
    }
    expect(scheduled.peakSRAMBytes <= [H16GTarget currentTarget].workingSetBytes,
           @"planned live set fits H16G SRAM budget");
    BOOL reused = NO;
    for (NSUInteger i = 0; i < activations.count; ++i)
        for (NSUInteger j = i + 1; j < activations.count; ++j)
            if (activations[i].sramOffset == activations[j].sramOffset &&
                activations[i].lastTask < activations[j].firstTask) reused = YES;
    expect(reused, @"non-overlapping activation lifetimes reuse an SRAM slot");
}

static void testLayoutPlansCarryShapeDerivedPacketStrategy(void) {
    NSString *source =
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 32, 128, 128]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(4)];\n"
         "    tensor<fp16, [1, 512, 32, 32]> packed = "
         "space_to_depth(x = x, block_size = b)[name = string(\"pack\")];\n"
         "    tensor<fp16, [1, 32, 128, 128]> y = "
         "depth_to_space(x = packed, block_size = b)[name = string(\"unpack\")];\n"
         "  } -> (y);\n}\n";
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = graphFromSource(source, diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    expect(scheduled != nil && diagnostics.errorCount == 0 &&
           scheduled.tasks.count == 2,
           @"S2D/D2S round trip becomes two ordinary scheduled layout tasks");
    if (!scheduled || scheduled.tasks.count != 2) return;
    ANETilePlan *pack = scheduled.tasks[0].tilePlan;
    ANETilePlan *unpack = scheduled.tasks[1].tilePlan;
    expect(pack.strategy == ANETileStrategyLayoutDMA3 &&
           unpack.strategy == ANETileStrategyLayoutDMA3,
           @"packet family is selected from footprint and block rules");
    expect([pack.inputShape isEqualToArray:@[@1,@32,@128,@128]] &&
           [pack.outputShape isEqualToArray:@[@1,@512,@32,@32]] &&
           [unpack.inputShape isEqualToArray:@[@1,@512,@32,@32]] &&
           [unpack.outputShape isEqualToArray:@[@1,@32,@128,@128]],
           @"scheduled layout plans retain symbolic input and output geometry");
    expect(pack.descriptorCount == 3 && unpack.descriptorCount == 3,
           @"the measured small-footprint layout family carries three TDs");
}

static NSString *matmulSource(NSUInteger size) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu]> a, tensor<fp16, [1, %lu, %lu]> b) {\n"
         "    bool f = const()[name = string(\"f\"), val = bool(false)];\n"
         "    tensor<fp16, [1, %lu, %lu]> y = matmul(transpose_x = f, transpose_y = f, x = a, y = b)[name = string(\"mm\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size];
}

static void testMatmulPlannerUsesDecodedRowFamilies(void) {
    NSArray<NSNumber *> *sizes = @[@512,@768,@2176];
    NSArray<NSNumber *> *rows = @[@512,@256,@128];
    NSArray<NSNumber *> *tiles = @[@1,@3,@17];
    for (NSUInteger index = 0; index < sizes.count; ++index) {
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEOperationGraph *operations = graphFromSource(
            matmulSource(sizes[index].unsignedIntegerValue),diagnostics);
        ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
            target:[H16GTarget currentTarget] diagnostics:diagnostics];
        expect(scheduled != nil && diagnostics.errorCount == 0 &&
               scheduled.tasks.count == 1,
               @"standalone square matmul becomes one scheduled hardware task");
        if (!scheduled) continue;
        ANETilePlan *plan = scheduled.tasks.firstObject.tilePlan;
        expect(plan.rows == rows[index].unsignedIntegerValue &&
               plan.count == tiles[index].unsignedIntegerValue &&
               plan.strategy == (tiles[index].unsignedIntegerValue == 1
                    ? ANETileStrategyDirect : ANETileStrategyMatrixRows),
               @"matmul planner selects the decoded 256/128-row packet family");
        expect(commandCount(scheduled,ANEScheduledCommandKindDMALoad) == 2 &&
               commandCount(scheduled,ANEScheduledCommandKindCompute) == 1 &&
               commandCount(scheduled,ANEScheduledCommandKindDMAStore) == 1 &&
               scheduled.peakSRAMBytes == 0,
               @"matmul schedule has two external loads and no fake resident tensor");
    }
}

static void testScheduledStagesPreservePrimitiveSemantics(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = fixture(@"attention.mil", diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    expect(scheduled != nil && diagnostics.errorCount == 0,
           @"attention stages schedule successfully");
    if (!scheduled || scheduled.tasks.count != 10) return;
    ANEScheduledTask *centerExp = scheduled.tasks[5];
    expect(centerExp.topology == ANEScheduledTopologyDirect,
           @"ordinary fused primitive stages retain direct topology");
    expect(centerExp.stages.count == 2,
           @"the center and exponential operations remain distinct stages");
    if (centerExp.stages.count == 2) {
        expect([centerExp.stages[0].operationName isEqualToString:@"sub"] &&
               [centerExp.stages[1].operationName isEqualToString:@"exp"],
               @"stage order follows primitive dataflow");
        expect([centerExp.stages[0].outputIdentifier
                    isEqualToString:@"probabilities.centered"] &&
               [centerExp.stages[1].inputIdentifiers
                    containsObject:@"probabilities.centered"],
               @"stage inputs and outputs retain the internal bridge");
    }
}

static void testDependencyWavesFollowDataflow(void) {
    NSString *source =
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 1, 128, 128]> x) {\n"
         "    tensor<fp16, [1, 1, 128, 128]> a = relu(x = x)[name = string(\"a\")];\n"
         "    tensor<fp16, [1, 1, 128, 128]> b = relu(x = x)[name = string(\"b\")];\n"
         "    tensor<fp16, [1, 1, 128, 128]> y = add(x = a, y = b)[name = string(\"join\")];\n"
         "  } -> (y);\n}\n";
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = graphFromSource(source, diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    expect(scheduled != nil && diagnostics.errorCount == 0 &&
           scheduled.tasks.count == 3,
           @"fork and join graph produces three tasks");
    if (!scheduled || scheduled.tasks.count != 3) return;
    expect(scheduled.tasks[0].waveIndex == 0 &&
           scheduled.tasks[1].waveIndex == 0,
           @"independent tasks share the first dependency wave");
    expect(scheduled.tasks[2].waveIndex == 1 &&
           [scheduled.tasks[2].dependencies isEqualToArray:@[@0, @1]],
           @"join task follows both producer tasks in the next wave");
}

static void testCarriedSurfaceBlocksOverlappingReuse(void) {
    ANEScheduledSurface *carry = [[ANEScheduledSurface alloc]
        initWithIdentifier:@"state" role:ANEScheduledSurfaceRoleCarry
        elementType:ANEElementTypeFP16 shape:@[@1, @1, @8, @8]
        firstTask:0];
    [carry extendLifetimeThroughTask:3];
    ANEScheduledSurface *temporary = [[ANEScheduledSurface alloc]
        initWithIdentifier:@"temporary"
        role:ANEScheduledSurfaceRoleIntermediate
        elementType:ANEElementTypeFP16 shape:@[@1, @1, @8, @8]
        firstTask:2];
    [temporary extendLifetimeThroughTask:2];
    NSUInteger peak = [ANEMemoryPlanner allocateSurfaces:@[carry, temporary]
        target:[H16GTarget currentTarget]];
    expect(carry.sramAllocated && temporary.sramAllocated,
           @"carried and temporary surfaces are allocated in SRAM");
    expect(carry.sramOffset != temporary.sramOffset,
           @"temporary storage cannot reuse a live carried-state slot");
    expect(peak == carry.byteLength + temporary.byteLength,
           @"overlapping carried state contributes to the complete live set");
}

static ANEScheduledRegionPlan *planWithTopology(
    ANEScheduledGraph *graph, ANEScheduledTopology topology) {
    for (ANEScheduledRegionPlan *plan in graph.composedPlans)
        if (plan.topology == topology) return plan;
    return nil;
}

static void testMatmulGELUFormsOneStructuralPlan(void) {
    for (NSString *name in @[@"matmul_gelu_128.mil",
                             @"matmul_gelu_256.mil"]) {
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEOperationGraph *operations = fixture(name, diagnostics);
        ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
            target:[H16GTarget currentTarget] diagnostics:diagnostics];
        ANEScheduledRegionPlan *plan = planWithTopology(
            scheduled, ANEScheduledTopologyDirect);
        if (!plan) {
            fprintf(stderr, "matmul-GELU graph %s:\n%s", name.UTF8String,
                    operations.textualDescription.UTF8String);
            dumpTaskGroups(scheduled);
        }
        expect(scheduled != nil && diagnostics.errorCount == 0,
               [name stringByAppendingString:@" schedules"]);
        expect(plan != nil && [plan.taskIndexes isEqualToArray:@[@0, @1]],
               @"matmul, reshape, and GELU form one direct composed plan");
        expect([plan.stageIdentifiers isEqualToArray:
                @[@"product", @"matrix", @"y"]],
               @"matmul-GELU plan retains structural stage order");
    }
}

static void testAttentionIsRecognizedFromPrimitiveEdges(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = fixture(@"attention.mil", diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    ANEScheduledRegionPlan *plan = planWithTopology(
        scheduled, ANEScheduledTopologyOnlineReduction);
    if (!plan) {
        fprintf(stderr, "attention candidate graph:\n%s",
                operations.textualDescription.UTF8String);
        dumpTaskGroups(scheduled);
    }
    expect(plan != nil, @"QK, softmax primitives, and PV form an online plan");
    expect([plan.stageIdentifiers containsObject:@"scores"] &&
           [plan.stageIdentifiers containsObject:@"probabilities"] &&
           [plan.stageIdentifiers containsObject:@"context"],
           @"online plan records score, normalization, and value stages");
    for (NSString *identifier in plan.stageIdentifiers)
        expect(![identifier containsString:@"attention"],
               @"online plan contains no workload-named stage");
}

static NSString *affineScanSource(BOOL exposeIntermediate) {
    NSString *returns = exposeIntermediate ? @"s0, y" : @"y";
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 1, 128, 128]> state, tensor<fp16, [1, 1, 128, 128]> a0, tensor<fp16, [1, 1, 128, 128]> b0, tensor<fp16, [1, 1, 128, 128]> a1, tensor<fp16, [1, 1, 128, 128]> b1) {\n"
         "    tensor<fp16, [1, 1, 128, 128]> p0 = mul(x = state, y = a0)[name = string(\"p0\")];\n"
         "    tensor<fp16, [1, 1, 128, 128]> s0 = add(x = p0, y = b0)[name = string(\"s0\")];\n"
         "    tensor<fp16, [1, 1, 128, 128]> p1 = mul(x = s0, y = a1)[name = string(\"p1\")];\n"
         "    tensor<fp16, [1, 1, 128, 128]> y = add(x = p1, y = b1)[name = string(\"s1\")];\n"
         "  } -> (%@);\n}\n", returns];
}

static void testAffineTransitionsFormAssociativeScan(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = graphFromSource(
        affineScanSource(NO), diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    ANEScheduledRegionPlan *plan = planWithTopology(
        scheduled, ANEScheduledTopologyAssociativeScan);
    if (!plan) {
        fprintf(stderr, "affine candidate graph:\n%s",
                operations.textualDescription.UTF8String);
        dumpTaskGroups(scheduled);
    }
    expect(plan != nil && plan.taskIndexes.count == 4,
           @"two affine transitions form one associative scan plan");
    expect([plan.stageIdentifiers isEqualToArray:@[@"p0", @"s0", @"p1", @"y"]],
           @"affine scan records both multiply-add transitions");
}

static void testEscapingAffineStateRejectsScanPlan(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = graphFromSource(
        affineScanSource(YES), diagnostics);
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics];
    expect(planWithTopology(scheduled,
               ANEScheduledTopologyAssociativeScan) == nil,
           @"an externally visible intermediate state blocks scan formation");
}

static NSString *sourceForFixture(NSString *name) {
    NSString *path = [@"tests/fixtures" stringByAppendingPathComponent:name];
    return [[NSString alloc] initWithData:[NSData dataWithContentsOfFile:path]
                                 encoding:NSUTF8StringEncoding];
}

static ANEScheduledGraph *scheduleSource(NSString *source) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *operations = graphFromSource(source, diagnostics);
    return operations ? [ANETaskScheduler scheduleGraph:operations
        target:[H16GTarget currentTarget] diagnostics:diagnostics] : nil;
}

static void testComposedPlansRejectInvalidForms(void) {
    NSString *attention = sourceForFixture(@"attention.mil");
    NSString *wrongScale = [attention stringByReplacingOccurrencesOfString:
        @"fp16(0.125)" withString:@"fp16(0.25)"];
    expect(planWithTopology(scheduleSource(wrongScale),
               ANEScheduledTopologyOnlineReduction) == nil,
           @"attention with a scale inconsistent with head width is rejected");

    NSString *escapingScores = [attention stringByReplacingOccurrencesOfString:
        @"} -> (y);" withString:@"} -> (scaled, y);"];
    expect(planWithTopology(scheduleSource(escapingScores),
               ANEScheduledTopologyOnlineReduction) == nil,
           @"attention score data that escapes the region blocks online lowering");

    NSString *fp32 = [sourceForFixture(@"matmul_gelu_128.mil")
        stringByReplacingOccurrencesOfString:@"fp16" withString:@"fp32"];
    expect(planWithTopology(scheduleSource(fp32),
               ANEScheduledTopologyDirect) == nil,
           @"matmul-GELU planning rejects a non-FP16 region");
}

static void testOnlineAttentionUsesBoundedTilesAndCarries(void) {
    NSArray<NSDictionary *> *cases = @[
        @{@"fixture": @"fa2_fp16_s128_d128.mil", @"tiles": @1},
        @{@"fixture": @"fa2_fp16_s256_d128.mil", @"tiles": @2},
    ];
    for (NSDictionary *testCase in cases) {
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEOperationGraph *operations = fixture(testCase[@"fixture"], diagnostics);
        ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:operations
            target:[H16GTarget currentTarget] diagnostics:diagnostics];
        ANEScheduledRegionPlan *plan = planWithTopology(
            scheduled, ANEScheduledTopologyOnlineReduction);
        NSUInteger expectedTiles = [testCase[@"tiles"] unsignedIntegerValue];
        expect(scheduled != nil && diagnostics.errorCount == 0,
               [testCase[@"fixture"] stringByAppendingString:@" schedules"]);
        expect(plan.tileRows == 128 &&
               plan.queryTileCount == expectedTiles &&
               plan.keyValueTileCount == expectedTiles,
               @"online attention records query and key/value tile counts");
        expect(plan.carriedSurfaceIdentifiers.count == 3,
               @"online attention carries row max, row sum and output state");
        expect([plan.boundaryInputIdentifiers
                   isEqualToArray:@[@"q", @"k", @"v"]] &&
               [plan.outputIdentifier isEqualToString:@"y"],
               @"online plan records its graph boundary in operand order");
        for (NSString *identifier in plan.carriedSurfaceIdentifiers)
            expect(surfaceNamed(scheduled, identifier).role ==
                       ANEScheduledSurfaceRoleCarry,
                   @"online state is represented by a carried surface");
        expect(surfaceNamed(scheduled, @"scores") == nil,
               @"online attention does not allocate the complete score tensor");
    }
}

int main(void) {
    @autoreleasepool {
        testConvReluBecomesOneScheduledComputeTask();
        testAttentionUsesInterTaskDMA();
        testW8A8PlannerKeepsFourConvModes();
        testSRAMAllocationsAreAlignedAndReused();
        testLayoutPlansCarryShapeDerivedPacketStrategy();
        testMatmulPlannerUsesDecodedRowFamilies();
        testScheduledStagesPreservePrimitiveSemantics();
        testDependencyWavesFollowDataflow();
        testCarriedSurfaceBlocksOverlappingReuse();
        testMatmulGELUFormsOneStructuralPlan();
        testAttentionIsRecognizedFromPrimitiveEdges();
        testAffineTransitionsFormAssociativeScan();
        testEscapingAffineStateRejectsScanPlan();
        testComposedPlansRejectInvalidForms();
        testOnlineAttentionUsesBoundedTilesAndCarries();
        printf("HWX planning: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
