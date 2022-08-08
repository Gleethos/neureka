package ut.backend.core


import neureka.backend.main.operations.other.Randomization
import spock.lang.Specification
import spock.lang.Subject

@Subject([Randomization])
class Randomization_Spec extends Specification {

    def 'Randomization is in essence the same algorithm as JDKs "Random".'()
    {
        given :
            var random = new Random()
            var seed = Randomization.initialScramble(666_42_666)
            var r2 = [0,0] as double[]

        when :
            random.seed = 666_42_666
        and :
            var r1 = random.nextGaussian()
            Randomization.gaussianFrom(seed, r2)

        then :
            r2[0] == r1
            r2[1] == random.nextGaussian()
    }

}
