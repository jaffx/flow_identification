#! /bin/bash
# 工程辅助命令
process_command=$1

show_xcmd() {
  grep -r "^#@xcmd" xcmd.sh | sed "s/#@xcmd //g" | sed "s/ /\t/1" | awk '{ printf "sh xcmd.sh %-20s\t%-20s\n",$1,$2}' | cat -n
  return 0
}

if [ -z "$process_command" ]; then
  show_xcmd
  exit 0
fi

printf "### 运行命令："

#@xcmd add 执行git add操作
add_list='Analysis README.md model requirements.txt run script conf lib xcmd.sh'
if [ "$process_command" = "add" ]; then
  echo "git add $add_list"
  echo "$add_list" | xargs git add

#@xcmd code_count 统计代码数量
elif [ "$process_command" = "code_count" ]; then
  echo "统计代码数量"
  python_file_count=$(find . -name "*.py" | wc -l)
  # shellcheck disable=SC2038
  code_line_count=$(find . -name "*.py" | xargs wc -l | sort -r | grep total | sed "s/total//g")
  printf ".py文件数量\t%s\n" "$python_file_count"
  printf ".py文件代码量\t%s\n" "$code_line_count"

#@xcmd clear_program 将路径下奇奇怪怪的文件删除
elif [ "$process_command" = "clear_program" ]; then
  echo "将路径下奇奇怪怪的文件(夹)删除"
  remove_files=' .DS_Store .idea __pycache__ '
  for file in $remove_files; do
    fileCount=$(find . '(' -name "$file" -path ./drops ')' | wc -l)
    echo "找到 $file 文件数量: $fileCount"
    find . -name $file -exec git rm -r --cache {} \;
    find . -name $file -exec rm -r {} \;
  done

#@xcmd move_result 将result文件夹保存到oss
elif [ "$process_command" = "move_result" ]; then
  echo "将result文件夹保存到oss"
  dt=$(date "+%Y%m%d_%H%M%S")
  zipFile="result_""$dt"".zip"
  cp -r result bk_result && zip -r -q "$zipFile" bk_result && oss cp "$zipFile" oss://result/ && rm "$zipFile" && rm -r bk_result

#@xcmd show_xcmd 展示支持的二级命令
elif [ "$process_command" = "show_xcmd" ]; then
  echo "xcmd命令列表"
  show_xcmd

# 未知命令
else
  echo "未知命令""${process_command}"
  show_xcmd
fi

# 判断运行结果
if [ $? -eq 0 ]; then
  echo "### 命令运行成功"
else
  echo "### 命令运行失败，结束码$?"
fi
