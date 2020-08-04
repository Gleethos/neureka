package ut.acceleration

import neureka.Neureka
import neureka.acceleration.Device
import neureka.acceleration.opencl.OpenCLDevice
import neureka.acceleration.opencl.utility.DeviceQuery
import spock.lang.Specification

class OpenCL_Unit_Test extends Specification
{

    def 'First found OpenCLDevice will have realistic properties.'()
    {
        given : 'This system supports OpenCL.'
            if ( !Neureka.instance().canAccessOpenCL() ) return

        when : 'Information about all existing OpenCL devices is being queried.'
            String query = DeviceQuery.query()

        then : 'The query string contains expected properties.'
        query.contains("DEVICE_NAME")
        query.contains("MAX_MEM_ALLOC_SIZE")
        query.contains("VENDOR")
        query.contains("CL_DEVICE_PREFERRED_VECTOR_WIDTH")
        query.contains("Info for device")
        query.contains("LOCAL_MEM_SIZE")
        query.contains("CL_DEVICE_TYPE")

        when : 'The first found Device instance is used.'
            OpenCLDevice cld = Device.find('first') as OpenCLDevice

        then : 'The device has realistic properties.'
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
