package it.acceleration

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
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
        device.add(s)

        then : 'An exception is being thrown.'
        def exception = thrown(IllegalStateException)

        and : 'It explains what went wrong.'
        exception.message=="Data parent is not outsourced!"
    }

}
