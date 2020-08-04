package it

import it.tests.CrossDeviceSlicedTensorIntegrationTest
import neureka.Neureka
import neureka.acceleration.Device
import neureka.acceleration.opencl.OpenCLPlatform
import spock.lang.Specification
import testutility.mock.DummyDevice


class Cross_Device_Sliced_Tensor_Test extends Specification
{

    def 'Cross device sliced tensor integration test runs without errors.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed = false
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            Device device = new DummyDevice()

        when : Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true)

        then : CrossDeviceSlicedTensorIntegrationTest.on(device, true)

        when : Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false)
        then : CrossDeviceSlicedTensorIntegrationTest.on(device, false)

        when :
            //=========================================================================
            if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
            //=========================================================================
            Device gpu = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0)

            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(true)
            OpenCLPlatform.PLATFORMS().get(0).recompile()

        then : CrossDeviceSlicedTensorIntegrationTest.on(gpu, true)

        when :
            Neureka.instance().settings().indexing().setIsUsingLegacyIndexing(false)
            OpenCLPlatform.PLATFORMS().get(0).recompile()

        then : CrossDeviceSlicedTensorIntegrationTest.on(gpu, false)

    }




}
