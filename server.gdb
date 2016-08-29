set print elements 0
set pagination off
set follow-fork-mode child
#break server.c:1823
#commands 1
#	p/x *(unsigned int*)out_pdu@1024
#	c
#end
#break server.c:1701

#break server.c:1718
#commands 1
#	p i
#	p *s
#	c
#end
#break init.c:1430

run --server=localhost

