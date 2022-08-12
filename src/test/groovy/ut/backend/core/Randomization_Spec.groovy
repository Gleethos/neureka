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


    def 'The Randomization class can fill various types of arrays with pseudo random numbers.'()
    {
        expect :
            Randomization.fillRandomly( new float[2], 42 ) == [-0.48528543, -1.4547276] as float[]
            Randomization.fillRandomly( new float[2], "I'm a seed!" ) == [-0.36105144, 0.09591457] as float[]
        and :
            Randomization.fillRandomly( new double[2], 42 ) == [-0.485285426832534, -1.454727674701364] as double[]
            Randomization.fillRandomly( new double[2], "I'm a seed!" ) == [-0.36105142873347185, 0.09591457295459412] as double[]
        and :
            Randomization.fillRandomly( new byte[2], 42 ) == [25, -68] as byte[]
            Randomization.fillRandomly( new byte[2], "I'm a seed!" ) == [-104, 26] as byte[]
        and :
            Randomization.fillRandomly( new int[2], 42 ) == [1335864601, 1769984444] as int[]
            Randomization.fillRandomly( new int[2], "I'm a seed!" ) == [-933496424, -1805080294] as int[]
        and :
            Randomization.fillRandomly( new long[2], 42 ) == [5737494774602678383, 7602025302221901325] as long[]
            Randomization.fillRandomly( new long[2], "I'm a seed!" ) == [-4009336612466399804, -7752760828057244539] as long[]
        and :
            Randomization.fillRandomly( new short[2], 42 ) == [-21223, -11844] as short[]
            Randomization.fillRandomly( new short[2], "I'm a seed!" ) == [-1640, -22246] as short[]
        and :
            Randomization.fillRandomly( new char[2], 42 ) == ['괙', '톼'] as char[]
            Randomization.fillRandomly( new char[2], "I'm a seed!" ) == ['輦', 'ꤚ'] as char[]
        and :
            Randomization.fillRandomly( new boolean[2], 42 ) == [false, false] as boolean[]
            Randomization.fillRandomly( new boolean[2], "I'm a seed!" ) == [true, true] as boolean[]
    }


}
