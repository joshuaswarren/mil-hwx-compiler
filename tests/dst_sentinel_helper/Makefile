LIBANE_DIR := $(realpath $(LIBANE_DIR))
CC := $(shell ls /usr/bin/clang++* 2>/dev/null | sort -V | tail -1)
ifeq ($(CC),)
CC := $(shell ls /usr/bin/clang++* 2>/dev/null | head -1)
endif
CFLAGS := -shared -fPIC -O2 -Wall -I$(LIBANE_DIR) -I.

all: libdst_prefill_test.so

libdst_prefill_test.so: dst_prefill_helper.c $(LIBANE_DIR)/ane.c $(LIBANE_DIR)/ane.h
	$(CC) $(CFLAGS) -o $@ dst_prefill_helper.c $(LIBANE_DIR)/ane.c

clean:
	rm -f libdst_prefill_test.so
