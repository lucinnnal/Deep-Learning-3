import numpy as np

class Variable:
    def __init__(self, data):
        # Variable에 numpy 인스턴스만 들어가도록 설정
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}는 지원하지 않습니다.')

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # 맨 마지막 출력값 gradient를 backward 메서드 안에서 계산
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  
            x, y = f.input, f.output  
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y)) # 0차원 array에 대한 계산은 np.float64로 변환됨. 그래서 np.float64로 나올 수도 있는 출력 값을 as_array 함수에 넣어 항상 Variable의 data가 np.array 인스턴스를 가질 수 있도록 함. 
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 함수 인스턴스 호출과 동시에 __call__ 메서드 호출하여서 함수 ouput값 뽑아내기
def square(x):
    return Square()(x) 

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    else:
        return x

if __name__ == "__main__":
    # Forward
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    # Backward
    y.backward()
    print(x.grad)