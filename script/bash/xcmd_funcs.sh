show_xcmd() {
  cat xcmd.sh | grep "^#@xcmd" xcmd.sh | sed "s/#@xcmd //g" | sed "s/ /\t/1" | awk '{ printf "sh xcmd.sh %-20s\t%-20s\n",$1,$2}' | cat -n
  return 0
}

clear_program() {
  remove_files=' .DS_Store .idea __pycache__ '
  for file in $remove_files; do
    fileCount=$(find . '(' -name "$file" -path ./drops ')' | wc -l)
    echo "找到 $file 文件数量: $fileCount"
    find . -name $file -exec git rm -r --cache {} \;
    find . -name $file -exec rm -r {} \;
  done

}
