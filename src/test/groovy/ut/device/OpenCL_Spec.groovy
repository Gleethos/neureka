package ut.device

import neureka.common.composition.Component
import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.devices.opencl.CLContext
import neureka.devices.opencl.OpenCLDevice
import neureka.devices.opencl.utility.DeviceQuery
import neureka.framing.Relation
import spock.lang.IgnoreIf
import spock.lang.Specification

class OpenCL_Spec extends Specification
{

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'First found OpenCLDevice will have realistic properties inside summary query.'()
    {
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

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'First found OpenCLDevice will have realistic numeric properties.'()
    {
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

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'First found OpenCLDevice will have realistic text properties.'()
    {
        when :  'The first found Device instance is used.'
            OpenCLDevice cld = Device.find('first') as OpenCLDevice

        then : 'The device has realistic properties.'
            !cld.name().isBlank()
            !cld.vendor().isBlank()
            cld.type() != OpenCLDevice.Type.UNKNOWN
            !cld.toString().isBlank()
            !cld.version().isBlank()
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'An OpenCLDevice will throw an exception when trying to add a tensor whose "data parent" is not outsourced.'()
    {
        given: 'The first found OpenCLDevice instance.'
            Device device = Device.find('first')

        and : 'A mocked tensor that is not outsourced.'
            Tsr<?> t = Mock(Tsr) // Could be : Tsr.of([4, 3], 2)
            t.isOutsourced() >> false

        and : 'Another mocked tensor that represents a slice of the prior one.'
            Tsr<?> s = Mock(Tsr) // Could be : t[1..3, 1..2]

        and : 'A mocked relation between both tensors returned by the slice as component.'
            Relation r = Mock(Relation)
            s.has(Relation.class) >> true
            s.get(Relation.class) >> r
            r.findRootTensor() >> t

        when : 'We try to add the slice to the device.'
            device.store(s)

        then : 'This will simple trigger the attempt of the device to register itself as component.'
            1 * s.set({ it == device })

        when : 'If the tensor was not a mock it would then cause the following change request to be dispatched:'
            device.update(new Component.OwnerChangeRequest() {
                @Override Tsr<?> getOldOwner() { return null }
                @Override Tsr<?> getNewOwner() { return s }
                @Override boolean executeChange() { return true }
            })

        then : 'The device will now try to store the tensor throw an exception because the tensor has an illegal state...'
            def exception = thrown(IllegalStateException)

        and : 'It explains what went wrong.'
            exception.message == "Data parent is not outsourced!"
    }


    def 'A given OpenCL context can be disposed!'() {

        given :
            CLContext context
            List<OpenCLDevice> devices = []
            Runnable dispose = {
                context = Neureka.get().backend().get(CLContext)
                assert context.platforms.size() > 0
                context.platforms.each {
                    assert it.devices.size() > 0
                    devices.addAll(it.devices)
                }
                context.dispose()
            }
            def thread = new Thread(dispose)

        when :
            thread.start()
            thread.join()

        then :
            noExceptionThrown()
        and :
            context.platforms.size() == 0
        and :
            devices.every {it.size() == 0}

    }


}
