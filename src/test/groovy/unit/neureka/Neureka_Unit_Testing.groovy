package unit.neureka

import neureka.Neureka
import spock.lang.Specification

class Neureka_Unit_Testing extends Specification
{
    void 'Neureka class instance has expected behaviour.'()
    {
        when : Neureka.instance().reset()
        then :
            assert !Neureka.instance().settings().isLocked()
            assert !Neureka.instance().settings().indexing().isUsingLegacyIndexing()
            assert !Neureka.instance().settings().debug().isKeepingDerivativeTargetPayloads()
            assert Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed()

        when : Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed = false
        then :
            assert !Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed()
            assert Neureka.instance().settings().autograd().isRetainingPendingErrorForJITProp()
            assert Neureka.version()=="0.2.4-pre"//version
    }



}
