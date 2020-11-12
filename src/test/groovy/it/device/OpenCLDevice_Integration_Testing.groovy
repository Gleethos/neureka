package it.device

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import spock.lang.Specification

class OpenCLDevice_Integration_Testing extends Specification
{

    def 'An OpenCLDevice will throw an exception when trying to add a tensor whose "data parent" is not outsourced.'()
    {
        given: 'This system supports OpenCL.'
            if (!Neureka.instance().canAccessOpenCL()) return
        and : 'The first found OpenCLDevice instance.'
            Device device = Device.find('first')
        and : 'A tensor and a slice tensor of the prior.'
            Tsr t = new Tsr([4, 3], 2)
            Tsr s = t[1..3, 1..2]

        when : 'We try to add the slice to the device.'
            device.store(s)

        then : 'An exception is being thrown.'
            def exception = thrown(IllegalStateException)

        and : 'It explains what went wrong.'
            exception.message=="Data parent is not outsourced!"
    }


    def 'The "getValue()" method of an outsourced tensor will return the expected array type.'()
    {
        given : 'This system supports OpenCL'
            if (!Neureka.instance().canAccessOpenCL()) return
        and : 'A new tensor belonging to the first found OpenCLDevice instance.'
            Tsr t = new Tsr([1, 2]).set(Device.find('first'))

        when : 'The tensor value is being fetched...'
            def value = t.getValue()

        then : 'This value object is an instance of a "double[]" array.'
            value instanceof double[]

        //when : 'The tensor datatype is being change from double to float...'
        //    t.to32()
        //    data = t.getValue()
        //then : 'The data that will be returned by the tensor is of type "float[]".'
        //    value instanceof float[] // WIP!
    }

    def 'The "getData()" method of an outsourced tensor will return null when outsourced.'()
    {
        given : 'This system supports OpenCL'
            if ( !Neureka.instance().canAccessOpenCL() ) return
        and : 'A new tensor belonging to the first found OpenCLDevice instance.'
            Tsr t = new Tsr([1, 2])

        expect : 'The tensor start with having data stored within.'
            t.data != null

        when : 'The tensor is being stored on the device...'
            t.set(Device.find('first'))
        and : 'The tensor value is being fetched...'
            def data = t.getData()

        then : 'The value variable is null.'
            data == null
    }

}
