#! /bin/zsh
# 工程辅助命令
process_command=$1

if [ -z "$process_command" ]; then
  echo "### 无命令"
  exit 1
fi

printf "### 运行命令："

# add 执行git add操作
add_list='Analysis README.md model requirements.txt run script conf tools xcmd.sh'
if [ "$process_command" = "add" ]; then
  echo "git add $add_list"
  echo "$add_list" | xargs git add

# code_count 统计代码数量
elif [ "$process_command" = "code_count" ]; then
  echo "统计代码数量"
  python_file_count=$(find . -name "*.py" | wc -l)
  # shellcheck disable=SC2038
  code_line_count=$(find . -name "*.py" | xargs wc -l | sort -r | grep total | sed "s/total//")
  printf ".py文件数量\t%d\n" "$python_file_count"
  printf ".py文件代码量\t%d\n" "$code_line_count"

# clear_program 将路径下奇奇怪怪的文件删除
elif [ "$process_command" = "clear_program" ]; then
  echo "将路径下奇奇怪怪的文件(夹)删除"
  remove_files=' .DS_Store .idea __pycache__ '
  for file in $remove_files; do
    fileCount=$(find . -name "$file" | wc -l)
    echo "找到 $file 文件数量: $fileCount"
    find . -name $file -exec "mv {} drops && gir rm -r {}" \;
  done

elif [ "$process_command" = "move_result" ]; then
  echo "将result文件夹保存到oss"
  dt=$(date "+%Y%m%d_%H%M%S")
  zipFile="result_""$dt"".zip"
  cp -r result bk_result && zip -r -q "$zipFile" bk_result && oss cp "$zipFile" oss://result/
  rm "$zipFile" && rm -r bk_result

# 未知命令
else
  echo "未知命令""${process_command}"
fi

# 判断运行结果
if [[ $? == 0 ]]; then
  echo " ### 命令运行成功"
else
  echo "### 命令运行失败，结束码$?"
fi
