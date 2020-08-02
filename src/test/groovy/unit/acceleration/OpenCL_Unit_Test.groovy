package unit.acceleration

import neureka.acceleration.Device
import neureka.acceleration.opencl.OpenCLDevice
import neureka.acceleration.opencl.utility.DeviceQuery
import spock.lang.Specification

class OpenCL_Unit_Test extends Specification
{

    def 'First found OpenCLDevice will have realistic properties.'()
    {
        given :
        //=========================================================================
        if(!System.getProperty("os.name").toLowerCase().contains("windows")) return
        //=========================================================================
        Device gpu = Device.find('first')

        when : String query = DeviceQuery.query()
        then :
        query.contains("DEVICE_NAME")
        query.contains("MAX_MEM_ALLOC_SIZE")
        query.contains("VENDOR")
        query.contains("CL_DEVICE_PREFERRED_VECTOR_WIDTH")
        query.contains("Info for device")
        query.contains("LOCAL_MEM_SIZE")
        query.contains("CL_DEVICE_TYPE")

        when : OpenCLDevice cld = (OpenCLDevice) gpu
        then :
        cld.globalMemSize()>1000
        !cld.name().equals("")
        cld.image2DMaxHeight()>100
        cld.image3DMaxHeight()>100
        cld.maxClockFrequenzy()>100
        !cld.vendor().equals("")
        !cld.toString().equals("")
        cld.maxConstantBufferSize()>1000
        cld.maxWriteImageArgs()>1
    }

}
