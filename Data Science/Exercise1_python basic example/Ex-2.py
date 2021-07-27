K=('Korean', 'Mathematics', 'English')
V=(90.3, 85.5, 92.7)
def makeDict(tu1,tu2):
    if len(tu1)==len(tu2):
        D=dict(zip(tu1,tu2))
    return D
    
print(makeDict(K,V))
