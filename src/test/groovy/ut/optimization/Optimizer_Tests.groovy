package ut.optimization

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function

//import org.junit.Test
import spock.lang.Specification

class Optimizer_Tests extends Specification{

    // WIP! : not yet completed!
    def 'NN calculations are being optimized'()
    {
        given :
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
        //def Y = data[0..-1, 3   ]

        def w1 = new Tsr([8, 3], ":-)").setRqsGradient(true)
        def w2 = new Tsr([7, 8], "O.o").setRqsGradient(true)
        def w3 = new Tsr([1, 7], ":P").setRqsGradient(true)

        def f = Function.create("softplus(I[0])")
        //def abs = Function.create("abs(I[0])")
        //def dox = Function.create("I[0]xI[1]")

        when :
        def y = w3.dot(f(w2.dot(f(w1.dot(X)))))
        //dox(new Tsr[]{abs(y-Y), new Tsr(y.shape(), 1,)}) // TODO!

        then:
            y.toString().contains(
                    "(6x3):[-0.72226E0, 8.09081E0, 0.53085E0, 13.9085E0, -0.76121E0, 92.1329E0, 73.4495E0, " +
                            "93.7233E0, -3.23894E0, -0.31264E0, -1.49487E0, 82.8246E0, -0.19767E0, 21.0283E0, " +
                            "-0.34664E0, -1.4782E0, -2.6672E0, 4.5934E0]"
            )


    }


}
