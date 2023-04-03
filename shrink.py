# shrink.py: a minimal C compressor
# Copyright (C) 2023 Nicholas Carlini.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import collections
import re

# TODO replace the unnecessary newlines

lines = open("prog.c").readlines()

if True:
    replace = []
    ls = []
    for line in lines:
        if '// MAKEINLINE' in line and False:
            line = line.strip()
            assert line.startswith("int")
            var, rest = line[3:].split("=")
            replace.append((var.strip(), rest.split(";")[0]))
        elif '// MAKETMP' in line and False:
            if 'TMP2' in line:
                newvar = 'tmp2'
            elif 'TMPP' in line:
                newvar = 'tmpp'
            else:
                newvar = 'tmp'
            line = line.strip()
            assert line.startswith("int")
            var, rest = line[3:].split(" = ")
            ls.append(newvar+ " = " + rest)
            replace.append((var.strip(), newvar))
        else:
            ls.append(line)
    print(replace)
    lines = "\n".join(ls)
    for src,dst in replace:
        lines = lines.replace(src,dst)
    lines = lines.split("\n")


open("/tmp/prog.c","w").write("\n".join(lines))
lines = [x.split("//")[0] for x in lines]

lines = [x.replace(";", "\x01")+"\x02" if '#' in x else x for x in lines]

lines = "\n".join(lines)
# Yes this would be better as a dictionary but once I got started...
lines = lines.replace("\\n","\x03")
lines = lines.replace('"r"',"\x04")
lines = lines.replace('%s',"\x05")
lines = lines.replace('%c',"\x06")
lines = lines.replace('%d',"\x07")
lines = lines.replace('malloc',"\x08")
lines = lines.replace('stdout',"\xe0")
lines = lines.replace('/*INT*/',"QQ0")
lines = lines.replace('fn',"out")
lines = lines.replace('opr',"result")
lines = lines.replace('reuse',"i")
lines = lines.replace('sub_cost',"k")
lines = lines.replace('raw',"trill")
lines = lines.replace('seq',"add_const")
lines = lines.replace('argv',"const1")
lines = lines.replace('merge',"alloc")
lines = lines.replace('sub_cost',"line")
lines = lines.replace('ptr',"dat")
lines = lines.replace('multiply_tile',"mt")
lines = lines.replace('add_tile',"at")
lines = lines.replace('dat',"i")
lines = lines.replace('rows',"j")
lines = lines.replace('cols',"k")
lines = lines.replace('ntok',"trill")
lines = lines.replace('best_i',"broadcast")

while '/*' in lines:
    first, _, rest = lines.partition("/*")
    drop, _, rest = rest.partition("*/")
    lines = first+rest

for _ in range(3):
    for _ in range(10):
        lines = lines.replace("\t", " ")
        lines = lines.replace("  ", " ")
    lines = lines.replace("\n ", "\n")
    for _ in range(10):
        lines = lines.replace("\n\n", "\n")
    
    for op in '+-*%^{}();?:,=<>|&[]/':
        for _ in range(10):
            lines = lines.replace(op+" ", op)
            lines = lines.replace(" "+op, op)
    for op in ':?,':
        for _ in range(10):
            lines = lines.replace(op+"\n", op)
            lines = lines.replace("\n"+op, op)
    
    for _ in range(3):
        lines = lines.split("\n")
        ls = []
        for x in lines:
            if x.startswith("int") and ls[-1].startswith("int") and x[-1] != '{' and ls[-1][-1] != '{':
                ls[-1] = ls[-1][:-1] # drop semicolon
                ls.append(","+x[3:])
            else:
                ls.append(x)
        lines = "\n".join(ls)
        lines = lines.replace("\n,", ",")
    lines = lines.replace("}", "} ")
    lines = lines.replace("{", "{ ")
    lines = lines.replace(";", ";\n")
    lines = lines.replace(" \n", "\n")
    lines = lines.replace("\n\n", "\n")

    # Remove single-line statement braces
    lines = lines.split("\n")
    for i in range(len(lines)-3):
        if len(lines[i]) and lines[i][-1] == '{' and lines[i+2] == '}':
            lines[i] = lines[i][:-1]
            lines[i+2] = ""
    lines = "\n".join(lines)

lines_no_quotes = re.sub(r'"[^"]*"', '', lines)
    
replace = collections.Counter(re.findall("[a-zA-Z_][a-zA-Z_0-9]*", lines_no_quotes))

del replace['h']
del replace['e']

print([x for x in replace if len(x) == 1])


for x in {'for', 'main', 'void', 'else', 'printf', 'if', 'include', 'sizeof', 'stdio', 'int', 'memcpy', 'string', 'while', 'return', 'atoi', 'getchar', 'memset', 'stdlib', 'long', 'define', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'malloc', 'typedef', 'FILE', 'tanh', 'exp', 'memcpy', 'scanf', 'struct', 'stdio', 'stdlib', 'string', 'float', 'fread', 'math', 'fscanf', 'fopen', 'char', 'sqrt', 'vocab', 'bpe', 'unsigned', 'strcat', 'fflush', 'stdout', 'omp', 'QQ0int', 'strlen', 'strcpy', 'strncmp', 'pragma', 'parallel', 'break', 'gets', 'ifdef', 'endif', 'fgets', 'stdin'}:
    del replace[x]

    
i = 0

chrs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
for x in replace:
    chrs = chrs.replace(x,"")

chrs = [x for x in chrs] + ["Q"+chr(0x41+n) for n in range(26)]
print(chrs)

replace = sorted(replace.items(), key=lambda x: (-len(x[0]), -x[1]))

print("Can't shorten", replace[len(chrs):])

rep = {}
for x,count in list(replace)[:len(chrs)]:
    if len(x) > len(chrs[i]):
        rep[x] = chrs[i]
        print("Replace", x, chrs[i])
        lines = lines.replace(x, chrs[i])
        i += 1

lines = lines.split("\n")
ls = []
for x in lines:
    if x.startswith("int") and '(' in x and ')' in x and x[-1] == '{':
        # function call
        ls.append(x.replace("int",""))
    else:
        ls.append(x)
lines = "\n".join(ls)

lines = lines.split("\n")
lines = "\n".join(lines)
lines = lines.replace("\n ", "\n")
lines = lines.replace("( ", "(")
lines = lines.replace(", ", ",")
lines = lines.replace(")\n", ")")
lines = lines.replace("(void*)", "")
lines = lines.split("\n")
lines = "\n".join(lines)

lines = lines.replace("\x01", ";")
lines = lines.replace("\x02", "\n")
lines = lines.replace("\x03", "\\n")
lines = lines.replace("\x04", '"r"')
lines = lines.replace("\x05", '%s')
lines = lines.replace("\x06", '%c')
lines = lines.replace("\x07", '%d')
lines = lines.replace(";}", "; }")
lines = lines.replace("\x08", 'malloc')
lines = lines.replace("\xe0", 'stdout')
lines = lines.replace("QQ0", 'int')
##
lines = lines.replace("\n\n", "\n")
lines = lines.replace("\n\n", "\n")
lines = lines.replace('Alice:"', 'Alice: "')
lines = lines.replace('%s:"', '%s: "')
lines = lines.replace("#define", "\n#define")


lines = lines.replace("main(", "main(int ")

print("LEN", len(lines))

open("/tmp/prog.c","w").write(lines)
