import re

string1 = "<span>hello world</span>"
rs = re.match("(?P<span_contexts><span>(.*?)</span>)", string1)
print(rs.groups())
print("group0", rs.group(0))
print("group1", rs.group(1))
print("group2", rs.group(2))
print(rs.group("span_contexts"))
