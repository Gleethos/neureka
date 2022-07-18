package ut.neureka

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphLock
import neureka.autograd.JITProp
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.calculus.Function
import neureka.common.utility.SettingsLoader
import neureka.devices.CustomDeviceCleaner
import neureka.devices.file.FileDevice
import neureka.devices.host.CPU
import neureka.devices.opencl.CLContext
import neureka.dtype.DataType
import neureka.framing.Relation
import neureka.view.NDPrintSettings
import spock.lang.*

import java.util.regex.Pattern

@Title('The Neureka context can be used and configured as expected.')
@Narrative('''

    This specification covers the behavior of the Neureka class which
    exposes a global API for configuring thread local contexts and library settings.
    The purpose of this is to assert that the API exposed by the Neureka class 
    is both thread local and configurable.
    This specification also exists to cover standards for the Neureka library in general.

''')
class Neureka_Spec extends Specification
{
    /**
     *  The below pattern defines a common standard for the strings returned by 'toString' methods
     *  returned by classes in the Neureka library.
     */
    @Shared
    Pattern toStringStandard = ~("^(" +
                                        "(([a-zA-Z]+\\.)*[A-Z][0-9a-zA-Z]+)" +
                                        "(@[0-9a-f]+)?" +
                                        "(#[0-9a-f]+)?" +
                                        "(" +
                                            "(\\[)(()|([a-zA-Z_\$][a-zA-Z_\$0-9]?)+=(.+))?(\\])" +
                                        ")?" +
                                ")\$")

    def setupSpec()
    {
        reportHeader """
                <h2> Neureka Behavior </h2>
                <br> 
                <p>
                    This specification covers the behavior
                    of the "Neureka" class, which is used as 
                    a thread local configuration.          
                </p>
            """
    }

    def setup() {
        // The following is similar to Neureka.get().reset() however it uses a groovy script for library settings:
        SettingsLoader.tryGroovyScriptsOn(Neureka.get(), script -> new GroovyShell(getClass().getClassLoader()).evaluate(script))
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }


    def 'Neureka class instance has expected behaviour.'()
    {
        expect : 'Important settings have their expected states.'
            !Neureka.get().settings().isLocked()
            !Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads()
            !Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed()

        when : 'Once something is changes to false...'
            Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed = true

        then : 'This setting change applies!'
            Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed()
            !Neureka.get().settings().autograd().isRetainingPendingErrorForJITProp()

        and : 'The version number is as expected!'
            Neureka.version()=="0.17.0"//version
    }

    def 'Neureka settings class can be locked.'(
            boolean value, def getter, def setter
    ) {
        given :
            def set = { it -> setter(Neureka.get().settings(), it) }
            def get = { getter(Neureka.get().settings()) }
        expect :
            get() == value

        when :
            Neureka.get().settings().setIsLocked(true)
        and :
            set(!value)
        then :
            get() == value

        when :
            Neureka.get().settings().setIsLocked(false)
        and :
            set(!value)
        then :
            get() != value

        cleanup :
            set(value)

        where :
            value | getter                                                                       | setter
            false | { Neureka.Settings it -> it.view().getNDPrintSettings().getIsLegacy()}       | { Neureka.Settings s, v -> s.view().getNDPrintSettings().setIsLegacy(v)}
            true  | { Neureka.Settings it -> it.view().getNDPrintSettings().getHasGradient()}    | { Neureka.Settings s, v -> s.view().getNDPrintSettings().setHasGradient(v)}
            false | { Neureka.Settings it -> it.view().getNDPrintSettings().getHasSlimNumbers()} | { Neureka.Settings s, v -> s.view().getNDPrintSettings().setHasSlimNumbers(v)}
            true  | { Neureka.Settings it -> it.view().getNDPrintSettings().getIsScientific()}   | { Neureka.Settings s, v -> s.view().getNDPrintSettings().setIsScientific(v)}
            false | { Neureka.Settings it -> it.ndim().isOnlyUsingDefaultNDConfiguration()}      | { Neureka.Settings s, v -> s.ndim().setIsOnlyUsingDefaultNDConfiguration(v)}
            false | { Neureka.Settings it -> it.debug().isKeepingDerivativeTargetPayloads()}     | { Neureka.Settings s, v -> s.debug().setIsKeepingDerivativeTargetPayloads(v)}
            true  | { Neureka.Settings it -> it.autograd().isPreventingInlineOperations()}       | { Neureka.Settings s, v -> s.autograd().setIsPreventingInlineOperations(v)}
            false | { Neureka.Settings it -> it.autograd().isRetainingPendingErrorForJITProp()}  | { Neureka.Settings s, v -> s.autograd().setIsRetainingPendingErrorForJITProp(v)}
            false | { Neureka.Settings it -> it.autograd().isApplyingGradientWhenTensorIsUsed()} | { Neureka.Settings s, v -> s.autograd().setIsApplyingGradientWhenTensorIsUsed(v)}
            false | { Neureka.Settings it -> it.autograd().isApplyingGradientWhenRequested()}    | { Neureka.Settings s, v -> s.autograd().setIsApplyingGradientWhenRequested(v)}
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

    @IgnoreIf({ data.neurekaObject == null })
    def 'Various library objects adhere to the same toString formatting convention!'(
            Object neurekaObject
    ) {
        expect : 'The provided object matches the following pattern defining a common standard!'
            toStringStandard.matcher(neurekaObject.toString()).matches()

        where : 'The following objects are being used..'
            neurekaObject << [
                    CPU.get(),
                    DataType.of(String),
                    new Relation<>(),
                    new JITProp<>([] as Set),
                    Neureka.get(),
                    Neureka.get().settings(),
                    Neureka.get().settings().autograd(),
                    Neureka.get().settings().debug(),
                    Neureka.get().settings().dtype(),
                    Neureka.get().settings().ndim(),
                    Neureka.get().settings().view(),
                    Neureka.get().backend().getAutogradFunction(),
                    Neureka.get().backend().getFunction(),
                    Neureka.get().backend(),
                    Neureka.get().backend().getFunctionCache(),
                    ExecutionCall.of(Tsr.of(3)).running(Neureka.get().backend().getOperation("+")).on(CPU.get()),
                    new CustomDeviceCleaner(),
                    (Tsr.of(2).setRqsGradient(true)*Tsr.of(-2)).graphNode,
                    new GraphLock(Function.of('i0*3/2')),
                    FileDevice.at('.')
            ]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() })
    def 'OpenCL related library objects adhere to the same toString formatting convention!'(
            Object neurekaCLObject
    ) {
        expect : 'The provided object matches the following pattern defining a common standard!'
            toStringStandard.matcher(neurekaCLObject.toString()).matches()

        where : 'The following objects are being used..'
            neurekaCLObject << [
                    Neureka.get().backend.get(CLContext),
                    Neureka.get().backend.get(CLContext).platforms[0],
                    Neureka.get().backend.get(CLContext).platforms[0].devices[0]
            ]
    }
    
    def 'Backend related library objects adhere to the same toString formatting convention!'(
            Operation operation
    ) {
        expect : 'The provided object matches the following pattern defining a common standard!'
            toStringStandard.matcher(operation.toString()).matches()
        and : 'The same criteria should also be met for every algorithm within the current operation.'
            operation.getAllAlgorithms().every {
                toStringStandard.matcher(it.toString()).matches()
            }

        where : 'The following operations are being used..'
            operation << Neureka.get().backend.operations

    }


}
