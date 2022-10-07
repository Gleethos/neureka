package it

import groovy.transform.CompileDynamic
import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.backend.ocl.CLBackend
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
        if ( Neureka.get().backend().has(CLBackend) )
            Neureka.get().backend().get(CLBackend).getSettings().autoConvertToFloat = false
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
        if ( Neureka.get().backend().has(CLBackend) )
            Neureka.get().backend().get(CLBackend).getSettings().autoConvertToFloat = true
    }

    @IgnoreIf({ data.deviceType == "GPU" && !Neureka.get().canAccessOpenCLDevice() })
    def 'Convolution can model matrix multiplications across devices.'(String deviceType) {
        given : 'A given device of any type and the settings configured for testing.'
            Device device = ( deviceType == "CPU" ) ? CPU.get() : Device.get('first')
            Neureka.get().reset()
            Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads = true
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)

        and : 'Two tensors, one requiring gradients and the other one does not.'
            var tensor1 = Tsr.of(new int[]{2, 2, 1}, new float[]{
                                                                    1,  2, //  3, 1,
                                                                    2, -3, // -2, -1,
                                                            })
                                                            .setRqsGradient( true )

            var tensor2 = Tsr.of(new int[]{1, 2, 2}, new float[]{
                                                                    -2, 3, //  0  7
                                                                    1, 2,  // -7  0
                                                            })
            device.store(tensor1).store(tensor2)

        and :
            Tsr product = Tsr.of("i0xi1", tensor1, tensor2)
            product.backward( Tsr.of(new int[]{2, 1, 2}, new float[]{1, 1, 1, 1}) )
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
            product.getMut().delete()
            tensor1.getMut().delete()
            //tensor2.delete() // TODO: FIX EXCEPTION!

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
            if ( Neureka.get().backend().has(CLBackend) )
                Neureka.get().backend().get(CLBackend).getSettings().autoConvertToFloat = true

        expect : 'The integration test runs successful.'
            CrossDeviceSystemTest.on(device)

        cleanup:
            if ( Neureka.get().backend().has(CLBackend) )
                Neureka.get().backend().get(CLBackend).getSettings().autoConvertToFloat = true

        where : 'The following settings are being used: '
            deviceType << ['CPU', 'GPU']
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null })
    def 'Test simple NN implementation with manual backprop'(Device device)
    {
        given:
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
            if ( Neureka.get().backend().has(CLBackend) )
                Neureka.get().backend().get(CLBackend).getSettings().autoConvertToFloat = true

        expect:
            device != null
        and :
            new SimpleNNSystemTest(SimpleNNSystemTest.Mode.CONVOLUTION).on(device)
        and:
            if ( !(device instanceof OpenCLDevice) )
                new SimpleNNSystemTest(SimpleNNSystemTest.Mode.MAT_MUL).on(device)

        cleanup:
            if ( Neureka.get().backend().has(CLBackend) )
                Neureka.get().backend().get(CLBackend).getSettings().autoConvertToFloat = false

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
            Tsr t = Tsr.of([2, 2], 4d).setRqsGradient(true).to(device)
        when : 'We now call the backward method on the tensor directly without having done any operations...'
            t.backward(1)
        and : 'Then we take the gradient to see what happened.'
            Tsr g = t.getGradient()

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
              Tsr<?> tensor, Device<?> device, Class<?> target, Function<?,?> lambda, String expected
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
            Tsr<?> result = tensor.mapTo(target, lambda)

        then : 'We expect the String representation of the tensor to be as expected!'
            result.toString() == expected
        and : 'We expect the result to have the expected target class!'
            result.itemType == target
        and : 'Lastly, the original tensor used as mapping source should be stored on the original device!'
            tensor.isOutsourced() == !(device instanceof CPU)
            tensor.device == device

        where : 'We use the following data to test this mapping for a wide range of types and values!'
            tensor                     | device               | target         | lambda   || expected
            Tsr.of(3.5)                | CPU.get()            | String.class   | {"~$it"} || '(1):[~3.5]'
            Tsr.of(3.5)                | Device.get('first')  | String.class   | {"~$it"} || '(1):[~3.5]'
            Tsr.ofFloats().scalar(3.5f)| CPU.get()            | String.class   | {"~$it"} || '(1):[~3.5]'
            Tsr.ofFloats().scalar(3.5f)| Device.get('first')  | String.class   | {"~$it"} || '(1):[~3.5]'
            Tsr.ofShorts().scalar(3.5f)| CPU.get()            | String.class   | {"~$it"} || '(1):[~3]'
            Tsr.ofShorts().scalar(3.5f)| Device.get('first')  | String.class   | {"~$it"} || '(1):[~3]'
            Tsr.ofBytes().scalar(2.7)  | CPU.get()            | String.class   | {"~$it"} || '(1):[~2]'
            Tsr.ofBytes().scalar(2.7)  | Device.get('first')  | String.class   | {"~$it"} || '(1):[~2]'
            Tsr.ofInts().scalar(6.1f)  | CPU.get()            | String.class   | {"~$it"} || '(1):[~6]'
            Tsr.ofInts().scalar(6.1f)  | Device.get('first')  | String.class   | {"~$it"} || '(1):[~6]'

            Tsr.of( 3.0 )              | Device.get('first')  | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.of(-1.0 )              | Device.get('first')  | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.of(0.5)                | Device.get('first')  | Integer.class  | {it*10}  || '(1):[5]'
            Tsr.of(0.7)                | Device.get('first')  | Long.class     | {it*5}   || '(1):[3]'
            Tsr.of(0.9)                | Device.get('first')  | Byte.class     | {it*2}   || '(1):[1]'
            Tsr.of(3.8)                | Device.get('first')  | Short.class    | {it/2}   || '(1):[1]'
            Tsr.of(3.0 )               | CPU.get()            | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.of(-1.0)               | CPU.get()            | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.of(0.5)                | CPU.get()            | Integer.class  | {it*10}  || '(1):[5]'
            Tsr.of(0.7)                | CPU.get()            | Long.class     | {it*5}   || '(1):[3]'
            Tsr.of(0.9)                | CPU.get()            | Byte.class     | {it*2}   || '(1):[1]'
            Tsr.of(3.8)                | CPU.get()            | Short.class    | {it/2}   || '(1):[1]'

            Tsr.ofFloats().scalar( 3f )| Device.get('first')  | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.ofFloats().scalar(-1f )| Device.get('first')  | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.ofFloats().scalar(0.5f)| Device.get('first')  | Integer.class  | {it*10}  || '(1):[5]'
            Tsr.ofFloats().scalar(0.7f)| Device.get('first')  | Long.class     | {it*5}   || '(1):[3]'
            Tsr.ofFloats().scalar(0.9f)| Device.get('first')  | Byte.class     | {it*2}   || '(1):[1]'
            Tsr.ofFloats().scalar(3.8f)| Device.get('first')  | Short.class    | {it/2}   || '(1):[1]'
            Tsr.ofFloats().scalar( 3f )| CPU.get()            | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.ofFloats().scalar(-1f )| CPU.get()            | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.ofFloats().scalar(0.5f)| CPU.get()            | Integer.class  | {it*10}  || '(1):[5]'
            Tsr.ofFloats().scalar(0.7f)| CPU.get()            | Long.class     | {it*5}   || '(1):[3]'
            Tsr.ofFloats().scalar(0.9f)| CPU.get()            | Byte.class     | {it*2}   || '(1):[1]'
            Tsr.ofFloats().scalar(3.8f)| CPU.get()            | Short.class    | {it/2}   || '(1):[1]'

            //Tsr.ofInts().scalar( 3 )   | Device.get('first') | Double.class   | {it*it} || '(1):[9.0]' // TODO: Allow for ints on the GPU
            //Tsr.ofInts().scalar(-1 )   | Device.get('first') | Float.class    | {it/2}  || '(1):[-0.5]'
            //Tsr.ofInts().scalar( 5 )   | Device.get('first') | Integer.class  | {it*10} || '(1):[50]'
            //Tsr.ofInts().scalar( 70)   | Device.get('first') | Long.class     | {it*5}  || '(1):[350]'
            //Tsr.ofInts().scalar( 90)   | Device.get('first') | Byte.class     | {it*2}  || '(1):[180]'
            //Tsr.ofInts().scalar( 37)   | Device.get('first') | Short.class    | {it/2}  || '(1):[18]'
            Tsr.ofInts().scalar( 3 )   | CPU.get()            | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.ofInts().scalar(-1 )   | CPU.get()            | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.ofInts().scalar( 5 )   | CPU.get()            | Integer.class  | {it*10}  || '(1):[50]'
            Tsr.ofInts().scalar( 70)   | CPU.get()            | Long.class     | {it*5}   || '(1):[350]'
            Tsr.ofInts().scalar( 90)   | CPU.get()            | Byte.class     | {it*2}   || '(1):[-76]'
            Tsr.ofInts().scalar( 37)   | CPU.get()            | Short.class    | {it/2}   || '(1):[18]'

            Tsr.ofShorts().scalar( 3 ) | Device.get('first')  | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.ofShorts().scalar(-1 ) | Device.get('first')  | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.ofShorts().scalar( 5 ) | Device.get('first')  | Integer.class  | {it*10}  || '(1):[50]'
            Tsr.ofShorts().scalar( 70) | Device.get('first')  | Long.class     | {it*5}   || '(1):[350]'
            Tsr.ofShorts().scalar( 90) | Device.get('first')  | Byte.class     | {it*2}   || '(1):[-76]'
            Tsr.ofShorts().scalar( 37) | Device.get('first')  | Short.class    | {it/2}   || '(1):[18]'
            Tsr.ofShorts().scalar( 3 ) | CPU.get()            | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.ofShorts().scalar(-1 ) | CPU.get()            | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.ofShorts().scalar( 5 ) | CPU.get()            | Integer.class  | {it*10}  || '(1):[50]'
            Tsr.ofShorts().scalar( 70) | CPU.get()            | Long.class     | {it*5}   || '(1):[350]'
            Tsr.ofShorts().scalar( 90) | CPU.get()            | Byte.class     | {it*2}   || '(1):[-76]'
            Tsr.ofShorts().scalar( 37) | CPU.get()            | Short.class    | {it/2}   || '(1):[18]'

            Tsr.ofBytes().scalar( 3 )  | Device.get('first') | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.ofBytes().scalar(-1 )  | Device.get('first') | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.ofBytes().scalar( 5 )  | Device.get('first') | Integer.class  | {it*10}  || '(1):[50]'
            Tsr.ofBytes().scalar( 70)  | Device.get('first') | Long.class     | {it*5}   || '(1):[350]'
            Tsr.ofBytes().scalar( 90)  | Device.get('first') | Byte.class     | {it*2}   || '(1):[-76]'
            Tsr.ofBytes().scalar( 37)  | Device.get('first') | Short.class    | {it/2}   || '(1):[18]'
            Tsr.ofBytes().scalar( 3 )  | CPU.get()           | Double.class   | {it*it}  || '(1):[9.0]'
            Tsr.ofBytes().scalar(-1 )  | CPU.get()           | Float.class    | {it/2}   || '(1):[-0.5]'
            Tsr.ofBytes().scalar( 5 )  | CPU.get()           | Integer.class  | {it*10}  || '(1):[50]'
            Tsr.ofBytes().scalar( 70)  | CPU.get()           | Long.class     | {it*5}   || '(1):[350]'
            Tsr.ofBytes().scalar( 90)  | CPU.get()           | Byte.class     | {it*2}   || '(1):[-76]'
            Tsr.ofBytes().scalar( 37)  | CPU.get()           | Short.class    | {it/2}   || '(1):[18]'

    }


}
