package ut.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.opencl.OpenCLDevice
import neureka.devices.opencl.utility.DeviceQuery
import neureka.framing.Relation
import spock.lang.Specification

class OpenCL_Unit_Test extends Specification
{

    def 'First found OpenCLDevice will have realistic properties inside summary query.'()
    {
        given: 'This system supports OpenCL.'
            if (!Neureka.instance().canAccessOpenCL()) return

        when: 'Information about all existing OpenCL devices is being queried.'
            String query = DeviceQuery.query()

        then: 'The query string contains expected properties.'
            query.contains("DEVICE_NAME")
            query.contains("MAX_MEM_ALLOC_SIZE")
            query.contains("VENDOR")
            query.contains("CL_DEVICE_PREFERRED_VECTOR_WIDTH")
            query.contains("Info for device")
            query.contains("LOCAL_MEM_SIZE")
            query.contains("CL_DEVICE_TYPE")
    }

    def 'First found OpenCLDevice will have realistic numeric properties.'()
    {
       given : 'This system supports OpenCL.'
            if ( !Neureka.instance().canAccessOpenCL() ) return

       when : 'The first found Device instance is used.'
            OpenCLDevice cld = Device.find('first') as OpenCLDevice

       then : 'The device has realistic properties.'
            cld.globalMemSize() > 1000
            cld.image2DMaxHeight() > 100
            cld.image2DMaxWidth() > 100
            cld.image3DMaxHeight() > 100
            cld.image3DMaxDepth() > 0
            cld.image3DMaxWidth() > 100
            cld.maxWorkGroupSize() > 10
            cld.maxClockFrequenzy() > 100
            cld.maxClockFrequenzy() > 100
            cld.maxConstantBufferSize() > 1000
            cld.maxWriteImageArgs() > 1
            cld.prefVecWidthChar() > 0
            cld.prefVecWidthDouble() > 0
            cld.prefVecWidthFloat() > 0
            cld.prefVecWidthInt() > 0
            cld.prefVecWidthLong() > 0
            cld.prefVecWidthShort() > 0
    }

    def 'First found OpenCLDevice will have realistic text properties.'()
    {
        given : 'This system supports OpenCL.'
            if ( !Neureka.instance().canAccessOpenCL() ) return

        when : 'The first found Device instance is used.'
            OpenCLDevice cld = Device.find('first') as OpenCLDevice

        then : 'The device has realistic properties.'
            !cld.name().isBlank()
            !cld.vendor().isBlank()
            !cld.type().isBlank()
            !cld.toString().isBlank()
            !cld.version().isBlank()
    }

    def 'An OpenCLDevice will throw an exception when trying to add a tensor whose "data parent" is not outsourced.'()
    {
        given: 'This system supports OpenCL.'
            if (!Neureka.instance().canAccessOpenCL()) return

        and : 'The first found OpenCLDevice instance.'
            Device device = Device.find('first')

        and : 'A mocked tensor that is not outsourced.'
            Tsr t = Mock(Tsr) // Could be : new Tsr([4, 3], 2)
            t.isOutsourced() >> false

        and : 'Another mocked tensor that represents a slice of the prior one.'
            Tsr s = Mock(Tsr) // Could be : t[1..3, 1..2]

        and : 'A mocked relation between both tensors returned by the slice as component.'
            Relation r = Mock(Relation)
            s.has(Relation.class) >> true
            s.find(Relation.class) >> r
            r.findRootTensor() >> t

        when : 'We try to add the slice to the device.'
            device.store(s)

        then : 'An exception is being thrown.'
            def exception = thrown(IllegalStateException)

        and : 'It explains what went wrong.'
            exception.message == "Data parent is not outsourced!"
    }


}
