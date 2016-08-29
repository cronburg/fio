set print elements 0
set pagination off
#set follow-fork-mode child

#break client.c:1191
#commands 1
#  printf "client:"
#  p *client
#  d 1
#  c
#end
#break client.c:1200
#commands 2
#  printf "cmd:"
#  p *cmd
#  c
#end

break server.c:521
break server.c:522
run --client=localhost --client=localhost,8766 ./fio.job
