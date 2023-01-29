import ast


def parseFile(filePath:str)->ast.AST:
  parsedTree: ast.AST
  with open(filePath, mode='r') as f:
    parsedText: str = f.read()
    parsedTree = ast.parse(parsedText)
  return parsedTree