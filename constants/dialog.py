from enum import Enum

class Role(Enum):
    USER = 'BUYER'
    SYSTEM  = 'SELLER'

intent_segment = '|'
sentence_segment = ","

if __name__ == "__main__":
    print(Role("A"))
    print(Role.USER in Role)
    print(Role('BUYER')==Role.USER)