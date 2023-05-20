package ut.tensors


import neureka.common.composition.Component
import neureka.Neureka
import neureka.Tensor
import neureka.devices.Device
import neureka.view.NDPrintSettings
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Tensors on Devices")
@Narrative('''

    This unit test specification covers 
    the expected behavior of tensors when interacting
    with instances of implementations of the Device interface.

''')
class Tensor_Device_Spec extends Specification
{
    def setupSpec() {
        reportHeader """
            Here you will find out how to store tensors on devices,
            how to move tensors between devices and how to use
            the device specific methods of the tensor class.
        """
    }

    def setup() {
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

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'Adding OpenCL device to tensor makes tensor be "outsourced" and contain the Device instance as component.'()
    {
        given : 'We get a device instance representing the GPU.'
            Device gpu = Device.get("gpu")
        and : 'We create a simple tensor.'
            Tensor t = Tensor.of([3, 4, 1], 3f)

        expect : 'The following is to be expected with respect to the given :'
            !t.has(Device.class)
            !t.isOutsourced()
            !gpu.has(t)

        when : 'The tensor is being added to the OpenCL device...'
            t.to(gpu)

        then : 'The now "outsourced" tensor has a reference to the device and vice versa!'
            t.has(Device.class)
            t.isOutsourced()
            gpu.has(t)
    }



    def 'Tensors try to migrate themselves to a device that is being added to them as component.'()
    {
        given : 'A mock device and a simple tensor instance.'
            def device = Mock(Device)
            var t = Tensor.of(2d)
            device.has(t) >>> [false, true, true]

        when : 'The mock device is being added to the tensor...'
            t.to(device)

        then : '...the tensor should not try to add itself to the given device via the "store" method.'
            0 * device.store(t)

        and : 'Instead it should use the "update" method, which is a formal callback from the internal component system...'
            1 * device.update({ it.type().name() == "ADDED" }) >> true

        and : 'It stores the device as a component.'
            t.has(Device.class)
    }

    def 'The device of a tensor can be accessed via the "device()" method.'()
    {
        given : 'A simple tensor having a device as component'
            def device = Mock(Device)
            device.has(_) >>> [false, true, true] // Some realistic return values to simulate tensor reception!
            var t = Tensor.of(1d)

        when :
            t.to(device)

        then :
            1 * device.update({
                Component.OwnerChangeRequest request -> request.executeChange()
            })

        when : 'The device is being accessed via the "device()" method...'
            Device found = t.get(Device.class)

        then : 'This found device should be the one that was set originally.'
            found == device

        and :
            found == t.getDevice()

        and :
            t.isOutsourced()
    }

    def 'When creating slices of tensors then this should trigger a "parent - child" relation noticeable to the device!'()
    {
        given : 'A 2D tensor having a device as component'
            def device = Mock(Device)
            var t = Tensor.of([3, 3],[1, 2, 3, 4, 5, 6, 7, 8, 9])
            device.has(t) >>> [false, true]

        when : 'A slice is being created from the given tensor...'
            var s = t[1..2, 0..2]

        and : 'The "parent tensor" is being migrated to the device...'
            t.to(device)

        then : '...this tensor should not try to add itself to the given device via the "store" method.'
            0 * device.store(t)

        and : 'Instead the "update" method should be called...'
            1 * device.update(_)

        and: 'Internally the Tensor may "asks" if it belongs to the Device. (before and after migration attempt)'
            (0.._) * device.has(t)
    }


}
