package ut.neureka

import neureka.Neureka
import spock.lang.Specification

class Neureka_Unit_Tests extends Specification
{
    void 'Neureka class instance has expected behaviour.'()
    {
        when : 'Neureka instance settings are being reset.'
            Neureka.instance().reset()
        then : 'Important settings have their expected states.'
            assert !Neureka.instance().settings().isLocked()
            assert !Neureka.instance().settings().indexing().isUsingLegacyIndexing()
            assert !Neureka.instance().settings().debug().isKeepingDerivativeTargetPayloads()
            assert Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed()

        when : 'One settings is changes to false...'
            Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed = false
        then : 'This setting change applies!'
            assert !Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed()
            assert Neureka.instance().settings().autograd().isRetainingPendingErrorForJITProp()
        and : 'The version number is as expected!'
            assert Neureka.version()=="0.2.4"//version
    }
    
}
