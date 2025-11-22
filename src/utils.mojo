""" 
Utilities module

Basic string manipulation helpers such as concatenation,
character wrapping, and byte-to-string conversion.
"""


def str_concat(s1: String, s2: String) -> String:
    return s1 + s2


def string_compare(a: String, b: String) -> Int:
    if a < b:
        return -1
    elif a > b:
        return 1
    return 0


def wrap(token: String) -> String:
    alias a = String("\\n")
    alias b = String("\\t")
    alias c = String("'")
    alias d = String('"')
    if token == a:
        return String("\n")
    if token == b:
        return String("\t")
    if token == c:
        return String("'")
    if token == d:
        return String('"')
    return token


def string_from_bytes(var bytes: List[UInt8]) -> String:
    var result = String("")
    for i in range(len(bytes)):
        result += chr(Int(bytes[i]))
    return result
