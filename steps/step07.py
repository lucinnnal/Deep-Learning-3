import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None # 변수 자신을 만들어낸 함수 (forward 연산을 하면서 자신을 만들어낸 creator를 지정하는 연산을 해줄 것임)

    def set_creator(self, func):
        self.creator = func
    
    def backward(self): #
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward() # 재귀적으로 backward 호출해서 자신의 creator와 creator의 input을 구해 만약 creator가 있다면 자신의 grad 속성을 creator backard에 전달해주어서 creator의 input에 대한 grad를 추출. 이를 creator가 없을 때까지 반복 => 처음 input x의 grad가 나올 때까지 반복

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # forward 연산 수행
        output = Variable(y)
        self.input = input # forward시에 들어온 input의 variable 지정
        output.set_creator(self) # 자신을 만든 함수를 self로 저장
        self.output = output # 출력도 저장

        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        return gy * 2 * self.input.data

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return gy * np.exp(self.input.data)

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))

    # Forward
    a = A(x)
    b = B(a)
    y = C(b)

    """
    # Backward
    y.grad = np.array(1.0)

    C = y.creator
    b = C.input
    b.grad = C.backward(y.grad)

    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)

    print(x.grad)
    """
    # Backward
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)