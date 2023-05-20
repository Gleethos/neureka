package it

import groovy.transform.CompileDynamic
import neureka.Data
import neureka.Neureka
import neureka.Shape
import neureka.Tensor
import neureka.backend.ocl.CLBackend
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import neureka.view.NDPrintSettings
import spock.lang.*
import st.tests.CrossDeviceSystemTest
import st.tests.SimpleNNSystemTest
import testutility.mock.DummyDevice

import java.util.function.Function

@Title("Cross Device Stress Test Specification")
@Narrative('''

    This specification is pretty much a system test which covers
    the behavior of the library as a whole across multiple devices!
    No matter which device is being used for a given stress test, the result should be the same...
                    
''')
@CompileDynamic
@Subject([Device])
class Cross_Device_Spec extends Specification
{
    def setup() {
        Neureka.get().backend().find(CLBackend).ifPresent{it.getSettings().autoConvertToFloat = false }
        Neureka.get().reset()
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

    def cleanup() {
        Neureka.get().backend.find(CLBackend).ifPresent { it.settings.autoConvertToFloat = true }
    }

    @IgnoreIf({ data.deviceType == "GPU" && !Neureka.get().canAccessOpenCLDevice() })
    def 'Convolution can model matrix multiplications across devices.'(String deviceType) {
        given : 'A given device of any type and the settings configured for testing.'
            Device device = ( deviceType == "CPU" ) ? CPU.get() : Device.get('first')
            Neureka.get().reset()
            Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads = true
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)

        and : 'Two tensors, one requiring gradients and the other one does not.'
            var tensor1 = Tensor.of(Shape.of(2, 2, 1),
                                            Data.of(
                                                1f,  2f, //  3, 1,
                                                2f, -3f, // -2, -1,
                                            ))
                                            .setRqsGradient( true )

            var tensor2 = Tensor.of(Shape.of(1, 2, 2),
                                            Data.of(
                                                -2f, 3f, //  0  7
                                                1f, 2f,  // -7  0
                                            ))
            device.store(tensor1).store(tensor2)

        and :
            Tensor product = Tensor.of("i0xi1", tensor1, tensor2)
            product.backward( Tensor.of(Shape.of(2, 1, 2), Data.of(1, 1, 1, 1)) )
            String result = product.toString({
                it.rowLimit = 15 // "rc"
                it.isScientific = false
                it.isMultiline = false
                it.hasGradient = false
                it.cellSize = 1
                it.hasValue = true
                it.hasRecursiveGraph = true
                it.hasDerivatives = false
                it.hasShape =  true
                it.isCellBound = false
                it.postfix = ""
                it.prefix = ""
                it.hasSlimNumbers = false
            })

        expect :
            result.contains(
                "[2x1x2]:(0.0, 7.0, -7.0, 0.0); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"
            )

        cleanup:
            product.mut.delete()
            tensor1.mut.delete()
            tensor2.mut.delete()

        where : 'The following settings are being used: '
            deviceType << ['CPU',  'GPU']
    }


    @IgnoreIf({ data.deviceType == "GPU" && !Neureka.get().canAccessOpenCLDevice() })
    def 'Cross device system test runs successfully.' (
            String deviceType
    ) {
        given : 'A given device of any type and the settings configured for testing.'
            Device device = ( deviceType == "CPU" ) ? CPU.get() : Device.get('first')
            Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads = true
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
            Neureka.get().backend.find(CLBackend).ifPresent { it.settings.autoConvertToFloat = true }

        expect : 'The integration test runs successful.'
            CrossDeviceSystemTest.on(device)

        cleanup:
            Neureka.get().backend.find(CLBackend).ifPresent { it.settings.autoConvertToFloat = true }

        where : 'The following settings are being used: '
            deviceType << ['CPU', 'GPU']
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null })
    def 'Test simple NN implementation with manual backprop'(Device device)
    {
        given:
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
            Neureka.get().backend.find(CLBackend).ifPresent { it.settings.autoConvertToFloat = true }

        expect:
            device != null
        and :
            new SimpleNNSystemTest(SimpleNNSystemTest.Mode.CONVOLUTION).on(device)
        and:
            if ( !(device instanceof OpenCLDevice) )
                new SimpleNNSystemTest(SimpleNNSystemTest.Mode.MAT_MUL).on(device)

        cleanup:
            Neureka.get().backend().find(CLBackend).ifPresent{ it.getSettings().autoConvertToFloat = false }

        where :
            device << [CPU.get(), Device.get('first gpu')]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null })
    def 'A gradient of ones can be set by calling the backward method on a tensor sitting on any device.'(
            Device device
    ) {
        // Some more asserts:
        given : 'We use the legacy representation of tensors for this little test!'
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
        and : 'We create a small matrix of 4 fours which requires a gradient and is stored on the provided device!'
            Tensor t = Tensor.of([2, 2], 4d).setRqsGradient(true).to(device)
        when : 'We now call the backward method on the tensor directly without having done any operations...'
            t.backward(1)
        and : 'Then we take the gradient to see what happened.'
            Tensor g = t.gradient.get()

        then : 'We expect this gradient to be all ones with the shape of our matrix!'
            g.toString().contains("[2x2]:(1.0, 1.0, 1.0, 1.0)")
            t.toString().contains("[2x2]:(4.0, 4.0, 4.0, 4.0):g:(1.0, 1.0, 1.0, 1.0)")
        and :
            t.isOutsourced() == !(device instanceof CPU)
            g.isOutsourced() == !(device instanceof CPU)
        and :
            t.device == device
            g.device == device

        where :
            device << [new DummyDevice(), Device.get('first gpu'), CPU.get()]
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null }) // We need to assure that this system supports OpenCL!
    def 'Mapping tensors works for every device (even if they are not used).'(
            Tensor<?> tensor, Device<?> device, Class<?> target, Function<?,?> lambda, String expected
    ) {
        given : 'We first make a note of the type we started with.'
            var originalType = tensor.itemType()

        when :  """
                    We start off by storing the provided tensor on the provided device.
                    This might be any kind of device like for example an $OpenCLDevice.
                    Which means the tensor might not be sitting in RAM!
                """
            tensor.to(device)

        then : 'After the tensor is stored on the device, we expect it to be still of the original type.'
            tensor.itemType == originalType

        when : """
                    We call the mapping method which is supposed to create a new tensor of the provided type.
                    This procedure is only supported when the tensor is stored in RAM, so when
                    the tensor is outsourced (stored on a device), then we expect that the mapping method
                    temporarily migrates the tensor back and forth internally...
               """
            Tensor<?> result = tensor.mapTo(target, lambda)

        then : 'We expect the String representation of the tensor to be as expected!'
            result.toString() == expected
        and : 'We expect the result to have the expected target class!'
            result.itemType == target
        and : 'Lastly, the original tensor used as mapping source should be stored on the original device!'
            tensor.isOutsourced() == !(device instanceof CPU)
            tensor.device == device

        where : 'We use the following data to test this mapping for a wide range of types and values!'
            tensor                     | device               | target         | lambda   || expected
            Tensor.of(3.5)                 | CPU.get()           | String.class  | {"~$it"} || '(1):[~3.5]'
            Tensor.of(3.5)                 | Device.get('first') | String.class  | {"~$it"} || '(1):[~3.5]'
            Tensor.ofFloats().scalar(3.5f) | CPU.get()           | String.class  | {"~$it"} || '(1):[~3.5]'
            Tensor.ofFloats().scalar(3.5f) | Device.get('first') | String.class  | {"~$it"} || '(1):[~3.5]'
            Tensor.ofShorts().scalar(3.5f) | CPU.get()           | String.class  | {"~$it"} || '(1):[~3]'
            Tensor.ofShorts().scalar(3.5f) | Device.get('first') | String.class  | {"~$it"} || '(1):[~3]'
            Tensor.ofBytes().scalar(2.7)   | CPU.get()           | String.class  | {"~$it"} || '(1):[~2]'
            Tensor.ofBytes().scalar(2.7)   | Device.get('first') | String.class  | {"~$it"} || '(1):[~2]'
            Tensor.ofInts().scalar(6.1f)   | CPU.get()           | String.class  | {"~$it"} || '(1):[~6]'
            Tensor.ofInts().scalar(6.1f)   | Device.get('first') | String.class  | {"~$it"} || '(1):[~6]'

            Tensor.of( 3.0 )               | Device.get('first') | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.of(-1.0 )               | Device.get('first') | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.of(0.5)                 | Device.get('first') | Integer.class | {it*10}  || '(1):[5]'
            Tensor.of(0.7)                 | Device.get('first') | Long.class    | {it*5}   || '(1):[3]'
            Tensor.of(0.9)                 | Device.get('first') | Byte.class    | {it*2}   || '(1):[1]'
            Tensor.of(3.8)                 | Device.get('first') | Short.class   | {it/2}   || '(1):[1]'
            Tensor.of(3.0 )                | CPU.get()           | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.of(-1.0)                | CPU.get()           | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.of(0.5)                 | CPU.get()           | Integer.class | {it*10}  || '(1):[5]'
            Tensor.of(0.7)                 | CPU.get()           | Long.class    | {it*5}   || '(1):[3]'
            Tensor.of(0.9)                 | CPU.get()           | Byte.class    | {it*2}   || '(1):[1]'
            Tensor.of(3.8)                 | CPU.get()           | Short.class   | {it/2}   || '(1):[1]'

            Tensor.ofFloats().scalar( 3f ) | Device.get('first') | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.ofFloats().scalar(-1f ) | Device.get('first') | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.ofFloats().scalar(0.5f) | Device.get('first') | Integer.class | {it*10}  || '(1):[5]'
            Tensor.ofFloats().scalar(0.7f) | Device.get('first') | Long.class    | {it*5}   || '(1):[3]'
            Tensor.ofFloats().scalar(0.9f) | Device.get('first') | Byte.class    | {it*2}   || '(1):[1]'
            Tensor.ofFloats().scalar(3.8f) | Device.get('first') | Short.class   | {it/2}   || '(1):[1]'
            Tensor.ofFloats().scalar( 3f ) | CPU.get()           | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.ofFloats().scalar(-1f ) | CPU.get()           | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.ofFloats().scalar(0.5f) | CPU.get()           | Integer.class | {it*10}  || '(1):[5]'
            Tensor.ofFloats().scalar(0.7f) | CPU.get()           | Long.class    | {it*5}   || '(1):[3]'
            Tensor.ofFloats().scalar(0.9f) | CPU.get()           | Byte.class    | {it*2}   || '(1):[1]'
            Tensor.ofFloats().scalar(3.8f) | CPU.get()           | Short.class   | {it/2}   || '(1):[1]'

            Tensor.ofInts().scalar( 3 )    | Device.get('first') | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.ofInts().scalar(-1 )    | Device.get('first') | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.ofInts().scalar( 5 )    | Device.get('first') | Integer.class | {it*10}  || '(1):[50]'
            Tensor.ofInts().scalar( 70)    | Device.get('first') | Long.class    | {it*5}   || '(1):[350]'
            Tensor.ofInts().scalar( 90)    | Device.get('first') | Byte.class    | {it*2}   || '(1):[-76]'
            Tensor.ofInts().scalar( 37)    | Device.get('first') | Short.class   | {it/2}   || '(1):[18]'
            Tensor.ofInts().scalar( 3 )    | CPU.get()           | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.ofInts().scalar(-1 )    | CPU.get()           | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.ofInts().scalar( 5 )    | CPU.get()           | Integer.class | {it*10}  || '(1):[50]'
            Tensor.ofInts().scalar( 70)    | CPU.get()           | Long.class    | {it*5}   || '(1):[350]'
            Tensor.ofInts().scalar( 90)    | CPU.get()           | Byte.class    | {it*2}   || '(1):[-76]'
            Tensor.ofInts().scalar( 37)    | CPU.get()           | Short.class   | {it/2}   || '(1):[18]'

            Tensor.ofShorts().scalar( 3 )  | Device.get('first') | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.ofShorts().scalar(-1 )  | Device.get('first') | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.ofShorts().scalar( 5 )  | Device.get('first') | Integer.class | {it*10}  || '(1):[50]'
            Tensor.ofShorts().scalar( 70)  | Device.get('first') | Long.class    | {it*5}   || '(1):[350]'
            Tensor.ofShorts().scalar( 90)  | Device.get('first') | Byte.class    | {it*2}   || '(1):[-76]'
            Tensor.ofShorts().scalar( 37)  | Device.get('first') | Short.class   | {it/2}   || '(1):[18]'
            Tensor.ofShorts().scalar( 3 )  | CPU.get()           | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.ofShorts().scalar(-1 )  | CPU.get()           | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.ofShorts().scalar( 5 )  | CPU.get()           | Integer.class | {it*10}  || '(1):[50]'
            Tensor.ofShorts().scalar( 70)  | CPU.get()           | Long.class    | {it*5}   || '(1):[350]'
            Tensor.ofShorts().scalar( 90)  | CPU.get()           | Byte.class    | {it*2}   || '(1):[-76]'
            Tensor.ofShorts().scalar( 37)  | CPU.get()           | Short.class   | {it/2}   || '(1):[18]'

            Tensor.ofBytes().scalar( 3 )   | Device.get('first') | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.ofBytes().scalar(-1 )   | Device.get('first') | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.ofBytes().scalar( 5 )   | Device.get('first') | Integer.class | {it*10}  || '(1):[50]'
            Tensor.ofBytes().scalar( 70)   | Device.get('first') | Long.class    | {it*5}   || '(1):[350]'
            Tensor.ofBytes().scalar( 90)   | Device.get('first') | Byte.class    | {it*2}   || '(1):[-76]'
            Tensor.ofBytes().scalar( 37)   | Device.get('first') | Short.class   | {it/2}   || '(1):[18]'
            Tensor.ofBytes().scalar( 3 )   | CPU.get()           | Double.class  | {it*it}  || '(1):[9.0]'
            Tensor.ofBytes().scalar(-1 )   | CPU.get()           | Float.class   | {it/2}   || '(1):[-0.5]'
            Tensor.ofBytes().scalar( 5 )   | CPU.get()           | Integer.class | {it*10}  || '(1):[50]'
            Tensor.ofBytes().scalar( 70)   | CPU.get()           | Long.class    | {it*5}   || '(1):[350]'
            Tensor.ofBytes().scalar( 90)   | CPU.get()           | Byte.class    | {it*2}   || '(1):[-76]'
            Tensor.ofBytes().scalar( 37)   | CPU.get()           | Short.class   | {it/2}   || '(1):[18]'
    }


}
