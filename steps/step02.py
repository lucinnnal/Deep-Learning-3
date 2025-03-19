import numpy as np
from step01 import Variable

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 구체적인 계산은 Function class의 forward 메서드에서 한다.
        return Variable(y)

    def forward(self, x):
        raise NotImplementedError # Function class를 상속 받아서 직접 forward 메서드를 정의하세요 !

class Square(Function): # Square class는 Function class를 상속받아서 forward 메서드 부분만 수정했다. (__call__ 메서드는 그대로 전송되었음)
    def forward(self, x):
        return x ** 2

if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)