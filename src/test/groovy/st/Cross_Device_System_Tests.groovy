package st

import groovy.transform.CompileDynamic
import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.HostCPU
import neureka.devices.opencl.CLContext
import spock.lang.IgnoreIf
import spock.lang.Specification
import st.tests.CrossDeviceSystemTest
import st.tests.SimpleNNSystemTest
import testutility.mock.DummyDevice

@CompileDynamic
class Cross_Device_System_Tests extends Specification
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



    def 'Test simple NN implementation with manual backprop'()
    {
        given :
            Neureka.get().settings().view().setIsUsingLegacyView(true)

        when : Device device = new DummyDevice()
        then : SimpleNNSystemTest.on(device)

        and :
        if ( !Neureka.get().canAccessOpenCL() ) return

        when : Device gpu = Neureka.get().context().get(CLContext.class).getPlatforms().get(0).getDevices().get(0)
        then : SimpleNNSystemTest.on(gpu)

        // Some more asserts:
        and : Tsr t = Tsr.of([2, 2], 4).setRqsGradient(true).to(gpu)
        when :
            t.backward(1)
            Tsr g = t.getGradient()

        then :
            assert g.toString().contains("[2x2]:(1.0, 1.0, 1.0, 1.0)")
            assert t.toString().contains("[2x2]:(4.0, 4.0, 4.0, 4.0):g:(1.0, 1.0, 1.0, 1.0)")
            assert t.isOutsourced()
            assert g.isOutsourced()
            //t.setIsOutsourced(false)
            //assert !g.isOutsourced()

    }





}
