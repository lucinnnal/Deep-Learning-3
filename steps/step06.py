import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 각 variable에 대한 grad 인스턴스 변수

class Function:
    def __call__(self, input):
        x = input.data
        out = self.forward(x)
        self.input = input # 함수에 들어온 input을 캐싱 (나중에 역전파 할 시에 grad 계산을 위해 저장) => call 함수를 호출 하여서 foward 연산할 시에 해당 input을 저장함

        return Variable(out) 

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy): # 이때 사용 되는 gy는 "upstream gradient" 
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data # function에서 forward 시에 사용한 call method에 저장된 input variable class의 data instance 변수 불러 오기
        gx = 2 * x * gy  # gy는 해당 함수의 upstream gradient
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data # forward할 때 저장해놓았던 self.input의 data instance 변수
        gx = np.exp(x) * gy # 해당 함수에 대한 input x의 gradient = upstream gradient * 현재 함수의 입력에 대한 미분값

        return gx 
    
if __name__ == "__main__":
    # Function instance create
    A = Square()
    B = Exp()
    C = Square()

    # Input Variable 
    x = Variable(np.array(0.5))

    # Forward
    a = A(x)
    b = B(a)
    y = C(b) 

    # Backward
    y.grad = np.array(1.0) #dy/dy # 역전파는 dy/dy = 1에서 시작
    b.grad = C.backward(y.grad) #dy/db
    a.grad = B.backward(b.grad) #db/da
    x.grad = A.backward(a.grad) #da/dx

    print(x.grad) # dy/dx