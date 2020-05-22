import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.optimization.Optimizer
import neureka.optimization.implementations.ADAM
import org.junit.Test

class OptimizerTests {


    @Test
    void test_ADAM()
    {
        Neureka.instance().reset()
        assert Function.create("I[0]*I[1]+(1-I[2])*I[3]").activate(new double[]{0.9, 0.0, 0.9, -3.0})==-0.29999999999999993
        assert Function.create("(1-I[0])*I[1]").activate(new double[]{0.9, -3.0})==-0.29999999999999993

        def t = new Tsr("( 1 - ", new Tsr(0.9), ") * ", new Tsr(-3))
        assert t.toString().equals("(1):[-0.29999E0]")
        t = new Tsr(new Tsr(0.9), "*", new Tsr(0), " + ( 1-", new Tsr(0.9), ") *", new Tsr(-3));
        assert t.toString().equals("(1):[-0.29999E0]")
        def mh = new Tsr(t, "/(1-", new Tsr(0.9), ")")
        assert mh.toString().equals("(1):[-3.0]")
        t = new Tsr(new Tsr(9),"^0.5+",new Tsr(1e-7))
        assert t.toString().contains("(1):[3.0]")
        t = new Tsr(new Tsr(0),"-",new Tsr(0.01),"*",mh,"/(",new Tsr(9.0),"^0.5+",new Tsr(1e-7),")")
        assert t.toString().equals("(1):[0.00999E0]")

        def grad = (int i)->new Tsr((i**3)%7-4)
        def expected = [
                0.009999999666666677, 0.02343838820965563, 0.030115667802083777, 0.040571568761377755, 0.04604647775568702, 0.04750863243746767, 0.054017821302644514, 0.06320597194500503, 0.07446842921800374, 0.08205723598079066
        ].collect(it -> new Tsr(it))

        Tsr w = new Tsr(0)
        Optimizer o = new ADAM(w)
        w.add(o)

        for(int i : 1..10){
            w.add(grad(i))
            w.applyGradient()
            assert w.toString().contains(expected[i-1].toString())
            assert w.shape().hashCode()==expected[i-1].shape().hashCode()
            assert w.translation().hashCode()==expected[i-1].translation().hashCode()
            assert w.idxmap().hashCode()==expected[i-1].idxmap().hashCode()
            assert w.spread().hashCode()==expected[i-1].spread().hashCode()
            assert w.offset().hashCode()==expected[i-1].offset().hashCode()
        }
    }




    @Test
    void test_optimized_NN(){
        Neureka.instance().reset()
        def data = new Tsr([8, 4], [ // a-b*c
                1, 2, 2, -3,
                3, -1, -1, 4,
                -1, -2, -3, -7,
                -2, -3, 4, 10,
                4, 5, -1, 9,
                6, 2, 3, 0,
                7, 3, 2, 1,
                -4, -4, 2, 4

        ])
        def X = data[0..-1, 0..2]
        def Y = data[0..-1, 3   ]

        def w1 = new Tsr([8, 3], ":-)").setRqsGradient(true)
        def w2 = new Tsr([7, 8], "O.o").setRqsGradient(true)
        def w3 = new Tsr([1, 7], ":P").setRqsGradient(true)

        def f = Function.create("softplus(I[0])")
        def abs = Function.create("abs(I[0])")
        def dox = Function.create("I[0]xI[1]")

        def y = w3.dot(f(w2.dot(f(w1.dot(X)))))

        assert y.toString().contains("(1x1x1x6x3):[-0.72226E0, 8.09081E0, 0.53085E0, 13.9085E0, -0.76121E0, 92.1329E0, 73.4495E0, 93.7233E0, -3.23894E0, -0.31264E0, -1.49487E0, 82.8246E0, -0.19767E0, 21.0283E0, -0.34664E0, -1.4782E0, -2.6672E0, 4.5934E0]")

        //dox(new Tsr[]{abs(y-Y), new Tsr(y.shape(), 1,)})

    }




}
