show_xcmd() {
  cat xcmd.sh | grep "^#@xcmd" xcmd.sh | sed "s/#@xcmd //g" | sed "s/ /\t/1" | awk '{ printf "sh xcmd.sh %-20s\t%-20s\n",$1,$2}' | cat -n
  return 0
}
