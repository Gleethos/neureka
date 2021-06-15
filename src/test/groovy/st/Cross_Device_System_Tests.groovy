package st

import groovy.transform.CompileDynamic
import st.tests.CrossDeviceSystemTest
import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.host.HostCPU
import neureka.devices.opencl.OpenCLPlatform
import spock.lang.Specification
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

    /*  ! WIP !
    def '...'(
            String deviceType, boolean legacyIndexing
    ) {
        given : 'A given device of any type and the settings configured for testing.'
            if (
            deviceType == "GPU" && // OpenCL cannot run inside TravisCI ! :/
                    !Neureka.instance().canAccessOpenCL()
            ) return
            Device device = ( deviceType == "CPU" ) ? HostCPU.instance() : Device.find('first')
            Neureka.instance().reset()
            Neureka.instance().settings().debug().isKeepingDerivativeTargetPayloads = true
            Neureka.instance().settings().view().isUsingLegacyView = true

        and : 'The indexing mode set to "legacy"!'
            Neureka.instance().settings().indexing().isUsingLegacyIndexing = legacyIndexing
            if ( device instanceof OpenCLDevice ) OpenCLPlatform.PLATFORMS().get(0).recompile()

        and : 'Two tensors, one requiring gradients and the other one does not.'
            def tensor1 = Tsr.of(new int[]{2, 2, 1}, new double[]{
                    1,  2, //  3, 1,
                    2, -3, // -2, -1,
            }).setRqsGradient( true )
            def tensor2 = Tsr.of(new int[]{1, 2, 2}, new double[]{
                    -2, 3, //  0  7
                    1, 2,  // -7  0
            })
            device.add(tensor1).add(tensor2)

        and :
            Tsr product = Tsr.of(new Tsr[]{tensor1, tensor2}, "i0xi1")
            product.backward( Tsr.of(new int[]{2, 1, 2}, new double[]{1, 1, 1, 1}) )
            String result = product.toString("rc")
            String t1AsStr = Arrays.toString(
                    ((legacyIndexing) ? new double[]{-1.0, -1.0, 5.0, 5.0} : new double[]{1.0, 3.0, 1.0, 3.0})
            )

        expect :
            result.contains(
                        "[2x1x2]:(" +
                                ((legacyIndexing)?"4.0, -13.0, 5.0, -4.0":"0.0, 7.0, -7.0, 0.0")+
                                "); =>d|[ [1x2x2]:(-2.0, 3.0, 1.0, 2.0) ]|:t{ [2x2x1]:(1.0, 2.0, 2.0, -3.0) }"
            )
            Arrays.toString(tensor1.value64()) == t1AsStr


            //product.delete();


        where : 'The following settings are being used: '
        deviceType  ||  legacyIndexing
        //'CPU'     ||     false
        //'CPU'     ||     true
        'GPU'     ||     true
        'GPU'     ||     false
    }
    */

    def 'Test cross device integration with default and legacy indexing.' (
            String deviceType
    ) {
        given : 'A given device of any type and the settings configured for testing.'
            if (
                deviceType == "GPU" && // OpenCL cannot run inside TravisCI ! :/
                !Neureka.get().canAccessOpenCL()
            ) return
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

        when : Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0)
        then : SimpleNNSystemTest.on(gpu)

        // Some more asserts:
        and : Tsr t = Tsr.of([2, 2], 4).setRqsGradient(true).set(gpu)
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
