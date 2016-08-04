
all: test

t0: fio-logs.tgz
	tar -xvf fio-logs.tgz

test: t0
	./run-test

clean:
	rm -f t*.out t? t?.exp

