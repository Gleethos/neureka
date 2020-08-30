package st


import neureka.Neureka
import neureka.acceleration.Device
import neureka.acceleration.opencl.OpenCLPlatform
import spock.lang.Specification
import st.tests.CrossDeviceSlicedTensorSystemTest
import testutility.mock.DummyDevice


class Cross_Device_Sliced_Tensor_System_Test extends Specification
{

    def 'Cross device sliced tensor integration test runs without errors.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed = false
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            Device device = new DummyDevice()

        //when : Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true)
        //then : CrossDeviceSlicedTensorSystemTest.on(device, true)
        //when : Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false)
        //then : CrossDeviceSlicedTensorSystemTest.on(device, false)

        when :
            if ( !Neureka.instance().canAccessOpenCL() ) return
            Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0)

            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true)
            OpenCLPlatform.PLATFORMS().get(0).recompile()

        then : CrossDeviceSlicedTensorSystemTest.on(gpu, true)

        when :
            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false)
            OpenCLPlatform.PLATFORMS().get(0).recompile()

        then : CrossDeviceSlicedTensorSystemTest.on(gpu, false)

    }




}
