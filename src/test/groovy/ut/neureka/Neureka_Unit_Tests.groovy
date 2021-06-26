package ut.neureka

import neureka.Neureka
import neureka.utility.SettingsLoader
import spock.lang.Specification

class Neureka_Unit_Tests extends Specification
{
    def setupSpec()
    {
        reportHeader """
                <h2> Neureka Behavior </h2>
                <br> 
                <p>
                    This specification covers the behavior
                    of the "Neureka" class, shich is used as 
                    a thread local configuration.          
                </p>
            """
    }

    def setup() {
        // The following is similar to Neureka.get().reset() however it uses a groovy script for library settings:
        SettingsLoader.tryGroovyScriptsOn(Neureka.get(), script -> new GroovyShell(getClass().getClassLoader()).evaluate(script))
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }


    def 'Neureka class instance has expected behaviour.'()
    {
        expect : 'Important settings have their expected states.'
            assert !Neureka.get().settings().isLocked()
            assert !Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads()
            assert Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed()

        when : 'One settings is changes to false...'
            Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed = false

        then : 'This setting change applies!'
            assert !Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed()
            assert Neureka.get().settings().autograd().isRetainingPendingErrorForJITProp()

        and : 'The version number is as expected!'
            assert Neureka.version()=="0.6.0"//version
    }

    
    def 'Every Thread instance has their own Neureka instance.'()
    {
        given : 'A map containing entries for Neureka instances.'
            def map = ['instance 1':null, 'instance 2':null]

        when : 'Two newly instantiated tensors store their Neureka instances in the map.'
            def t1 = new Thread({ map['instance 1'] = Neureka.get() })
            def t2 = new Thread({ map['instance 2'] = Neureka.get() })

        and : 'The tensors are being started and joined.'
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        then : 'The map entries will no longer be filled with null.'
            map['instance 1'] != null
            map['instance 2'] != null

        and : 'The Neureka instances stored in the map will be different objects.'
            map['instance 1'] != map['instance 2']

    }


    
}
