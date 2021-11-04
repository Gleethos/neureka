package st

import groovy.transform.CompileDynamic
import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.HostCPU
import neureka.devices.opencl.OpenCLDevice
import spock.lang.IgnoreIf
import spock.lang.Specification
import st.tests.CrossDeviceSystemTest
import st.tests.SimpleNNSystemTest
import testutility.mock.DummyDevice

import java.util.function.Function

@CompileDynamic
class Cross_Device_Spec extends Specification
{
    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }


    @IgnoreIf({deviceType == "GPU" && !Neureka.get().canAccessOpenCL()})
    def 'Convolution can model matrix multiplications across devices.'(String deviceType) {
        given : 'A given device of any type and the settings configured for testing.'
            Device device = ( deviceType == "CPU" ) ? HostCPU.instance() : Device.find('first')
            Neureka.get().reset()
            Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads = true
            Neureka.get().settings().view().isUsingLegacyView = true

        and : 'Two tensors, one requiring gradients and the other one does not.'
            def tensor1 = Tsr.of(new int[]{2, 2, 1}, new double[]{
                    1,  2, //  3, 1,
                    2, -3, // -2, -1,
            }).setRqsGradient( true )
            def tensor2 = Tsr.of(new int[]{1, 2, 2}, new double[]{
                    -2, 3, //  0  7
                    1, 2,  // -7  0
            })
            device.store(tensor1).store(tensor2)

        and :
            Tsr product = Tsr.of("i0xi1", tensor1, tensor2)
            product.backward( Tsr.of(new int[]{2, 1, 2}, new double[]{1, 1, 1, 1}) )
            String result = product.toString("rc")


        expect :
            result.contains(
                        "[2x1x2]:(0.0, 7.0, -7.0, 0.0); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"
            )

        cleanup:
            product.delete()
            tensor1.delete()
            //tensor2.delete() // TODO: FIX EXCEPTION!


        where : 'The following settings are being used: '
            deviceType << ['CPU',  'GPU']
    }


    @IgnoreIf({deviceType == "GPU" && !Neureka.get().canAccessOpenCL()})
    def 'Test cross device integration with default and legacy indexing.' (
            String deviceType
    ) {
        given : 'A given device of any type and the settings configured for testing.'
            Device device = ( deviceType == "CPU" ) ? HostCPU.instance() : Device.find('first')
            Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads = true
            Neureka.get().settings().view().isUsingLegacyView = true

        expect : 'The integration test runs successful.'
            CrossDeviceSystemTest.on(device)

        where : 'The following settings are being used: '
            deviceType << ['CPU', 'GPU']
    }


    @IgnoreIf({!Neureka.get().canAccessOpenCL() && (device instanceof OpenCLDevice)})
    def 'Test simple NN implementation with manual backprop'(Device device) {
        given:
            Neureka.get().settings().view().setIsUsingLegacyView(true)

        expect:
            new SimpleNNSystemTest(SimpleNNSystemTest.Mode.CONVOLUTION).on(device)
        and:
            if ( !(device instanceof OpenCLDevice) )
                new SimpleNNSystemTest(SimpleNNSystemTest.Mode.MAT_MUL).on(device)

        where :
            device << [new DummyDevice(), Device.find('first gpu')]
    }

    @IgnoreIf({!Neureka.get().canAccessOpenCL() && (device instanceof OpenCLDevice)})
    def 'Test back-prop'(Device device)
    {
        // Some more asserts:
        given :
            Neureka.get().settings().view().setIsUsingLegacyView(true)
            Tsr t = Tsr.of([2, 2], 4).setRqsGradient(true).to(device)
        when :
            t.backward(1)
            Tsr g = t.getGradient()

        then :
            g.toString().contains("[2x2]:(1.0, 1.0, 1.0, 1.0)")
            t.toString().contains("[2x2]:(4.0, 4.0, 4.0, 4.0):g:(1.0, 1.0, 1.0, 1.0)")
            //t.isOutsourced()
            //g.isOutsourced()
            //t.setIsOutsourced(false)
            //!g.isOutsourced()

        where :
            device << [new DummyDevice(), Device.find('first gpu')]

    }



    @IgnoreIf({!Neureka.get().canAccessOpenCL() && (device instanceof OpenCLDevice)}) // We need to assure that this system supports OpenCL!
    def 'Mapping tensors works for every device (even if they are not used).'(
              def tensor, Device device, Class<?> target, Function<?,?> lambda, String expected
    ) {
        given :
            tensor.to(device)

        when :
            def result = tensor.mapTo(target, lambda)

        then :
            result.toString() == expected
        and :
            result.valueClass == target
            tensor.isOutsourced() == !(device instanceof HostCPU)
            tensor.device == device

        where :
            tensor                     | device               | target         | lambda  || expected
            Tsr.of( 3 )                | Device.find('first') | Double.class   | {it*it} || '(1):[9.0]'
            Tsr.of(-1 )                | Device.find('first') | Float.class    | {it/2}  || '(1):[-0.5]'
            Tsr.of(0.5)                | Device.find('first') | Integer.class  | {it*10} || '(1):[5.0]'
            Tsr.of(0.7)                | Device.find('first') | Long.class     | {it*5}  || '(1):[3.0]'
            Tsr.of(0.9)                | Device.find('first') | Byte.class     | {it*2}  || '(1):[1.0]'
            Tsr.of(3.8)                | Device.find('first') | Short.class    | {it/2}  || '(1):[1.0]'
            Tsr.of( 3 )                | HostCPU.instance()   | Double.class   | {it*it} || '(1):[9.0]'
            Tsr.of(-1 )                | HostCPU.instance()   | Float.class    | {it/2}  || '(1):[-0.5]'
            Tsr.of(0.5)                | HostCPU.instance()   | Integer.class  | {it*10} || '(1):[5.0]'
            Tsr.of(0.7)                | HostCPU.instance()   | Long.class     | {it*5}  || '(1):[3.0]'
            Tsr.of(3.8)                | HostCPU.instance()   | Short.class    | {it/2}  || '(1):[1.0]'

            Tsr.ofFloats().scalar( 3f )| Device.find('first') | Double.class   | {it*it} || '(1):[9.0]'
            Tsr.ofFloats().scalar(-1f )| Device.find('first') | Float.class    | {it/2}  || '(1):[-0.5]'
            Tsr.ofFloats().scalar(0.5f)| Device.find('first') | Integer.class  | {it*10} || '(1):[5.0]'
            Tsr.ofFloats().scalar(0.7f)| Device.find('first') | Long.class     | {it*5}  || '(1):[3.0]'
            Tsr.ofFloats().scalar(0.9f)| Device.find('first') | Byte.class     | {it*2}  || '(1):[1.0]'
            Tsr.ofFloats().scalar(3.8f)| Device.find('first') | Short.class    | {it/2}  || '(1):[1.0]'
            Tsr.ofFloats().scalar( 3f )| HostCPU.instance()   | Double.class   | {it*it} || '(1):[9.0]'
            Tsr.ofFloats().scalar(-1f )| HostCPU.instance()   | Float.class    | {it/2}  || '(1):[-0.5]'
            Tsr.ofFloats().scalar(0.5f)| HostCPU.instance()   | Integer.class  | {it*10} || '(1):[5.0]'
            Tsr.ofFloats().scalar(0.7f)| HostCPU.instance()   | Long.class     | {it*5}  || '(1):[3.0]'
            Tsr.ofFloats().scalar(3.8f)| HostCPU.instance()   | Short.class    | {it/2}  || '(1):[1.0]'

            //Tsr.ofInts().scalar( 3 )   | Device.find('first') | Double.class   | {it*it} || '(1):[9.0]' // TODO: Allow for ints on the GPU
            //Tsr.ofInts().scalar(-1 )   | Device.find('first') | Float.class    | {it/2}  || '(1):[-0.5]'
            //Tsr.ofInts().scalar( 5 )   | Device.find('first') | Integer.class  | {it*10} || '(1):[50.0]'
            //Tsr.ofInts().scalar( 70)   | Device.find('first') | Long.class     | {it*5}  || '(1):[350.0]'
            //Tsr.ofInts().scalar( 90)   | Device.find('first') | Byte.class     | {it*2}  || '(1):[45.0]'
            //Tsr.ofInts().scalar( 37)   | Device.find('first') | Short.class    | {it/2}  || '(1):[18.0]'
            Tsr.ofInts().scalar( 3 )   | HostCPU.instance()   | Double.class   | {it*it} || '(1):[9.0]'
            Tsr.ofInts().scalar(-1 )   | HostCPU.instance()   | Float.class    | {it/2}  || '(1):[-0.5]'
            Tsr.ofInts().scalar( 5 )   | HostCPU.instance()   | Integer.class  | {it*10} || '(1):[50.0]'
            Tsr.ofInts().scalar( 70)   | HostCPU.instance()   | Long.class     | {it*5}  || '(1):[350.0]'
            Tsr.ofInts().scalar( 90)   | HostCPU.instance()   | Short.class    | {it/2}  || '(1):[45.0]'
    }


}
