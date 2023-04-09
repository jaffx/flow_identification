#! /bin/zsh
# 工程辅助命令
process_command=$1

if [[ $process_command == "" ]]; then
  echo "### 无命令"
else
  echo -n "### 运行命令："

  # add 执行git add操作
  add_list='Analysis tools model script xcmd README.md requirements.txt train.py validation.py'
  if [[ $process_command == "add" ]]; then
    echo "git add $add_list"
    echo "$add_list" | xargs git add

  # code_count 统计代码数量

  elif [[ $process_command == "code_count" ]]; then
    echo "统计代码数量"
    pyfiles=$(find . -name "*.py")
    python_file_count=$(echo $pyfiles | wc -l | xargs echo)
    echo $pyfiles | xargs wc -l | xargs echo
    code_line_count=$(echo $pyfiles | xargs wc -l | xargs echo)
    print $code_line_count
    printf ".py文件数量\t%d\n" python_file_count
    printf ".py文件代码量\t%d\n" code_line_count
  # 未知命令
  else
    echo "未知命令""${process_command}"
  fi
fi
# 判断运行结果
if [[ $? == 0 ]]; then
  echo "### 命令运行成功"
else
  echo "### 命令运行失败，结束码$?"
fi
