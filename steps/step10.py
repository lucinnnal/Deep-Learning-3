from step09 import *
import unittest # 파이썬 단위 테스트 라이브러리

# linux 명령어 : python -m unittest step10.py를 통해 파이썬 파일을 테스트 모드로 실행. unittest가 실행되려면, 테스트 케이스가 unittest.TestCase를 상속받아 정의되어 있어야 함.
# python -m unittest discover test (test/ directory에서 test*.py 형식(default, 바꾸어줄 수 있음)의 모든 파일 실행)
# python -m unittest discover -s test -p "check_*.py" 
# -s test → 검색할 디렉터리 지정 (test/)
#-p "check_*.py" → 실행할 파일 패턴 변경 (기본값: test*.py)

# 아니면 main 함수에 unittest.main() 코드 추가

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data-eps)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected) # 같은지

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg) # True 인지
    
    # test 함수를 3개 정의했으므로 3개의 테스트를 진행