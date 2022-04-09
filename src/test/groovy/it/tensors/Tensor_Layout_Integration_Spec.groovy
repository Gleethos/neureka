package it.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.common.utility.SettingsLoader
import neureka.devices.Device
import neureka.ndim.config.NDConfiguration
import neureka.view.TsrStringSettings
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Row or Column Major. Why not both?")
@Narrative('''

    Although Neureka exposes tensors as row major tensors from 
    a users point of view, it does in fact support both row major and column major 
    based tensor layout under the hood.
    Here we cover how the layout of tensors can be modified
    and we ensure the different tensor types still work as expected...
    (The features in this specification involve mutating tensors, be careful when playing around with this yourself)
    
''')
class Tensor_Layout_Integration_Spec extends Specification
{
    def setup() {
        // The following is similar to Neureka.get().reset() however it uses a groovy script for library settings:
        SettingsLoader.tryGroovyScriptsOn(Neureka.get(), script -> new GroovyShell(getClass().getClassLoader()).evaluate(script))
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
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

    def 'A new transposed version of a given tensor will be returned by the "T()" method.'()
    {
        given : 'We want to view tensors in the "[shape]:(value)" format so we set the corresponding flag.'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
        and : 'We instantiate a test tensor:'
            var t = Tsr.of([2, 3], [
                    1d, 2d, 3d,
                    4d, 5d, 6d
            ])

        when : 'A two by three matrix is being transposed...'
            var t2 = t.T()

        then : 'The resulting tensor should look like this:'
            t2.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")

        when : 'We try the same operation with a column major tensor...'
            t2 = t.unsafe.toLayout(NDConfiguration.Layout.COLUMN_MAJOR).T

        then : 'Once again, the resulting tensor should look like this:'
            t2.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")
    }


    @IgnoreIf({ data.device == 'GPU' && !Neureka.get().canAccessOpenCL() })
    def 'Matrix multiplication works for both column and row major matrices across devices.'(
            String device, String expectedString
    ) {
        given : 'We want to view tensors in the "(shape:[value]" format so we set the corresponding flag.'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
        and :
            var a = Tsr.ofFloats().withShape(2, 3).andWhere({it, idx->((7**it)%11-5).floatValue()})
            var b = Tsr.ofFloats().withShape(3, 4).andWhere({it, idx->((5**it)%11-5).floatValue()})
            Device.find(device).store(a).store(b)
        expect :
            a.matMul(b).toString({it.hasSlimNumbers = true}) == expectedString

        when :
            a.unsafe.toLayout(NDConfiguration.Layout.COLUMN_MAJOR)
            b.unsafe.toLayout(NDConfiguration.Layout.COLUMN_MAJOR)
        then :
            a.matMul(b).toString({it.hasSlimNumbers = true}) == expectedString

        when :
            a.unsafe.toLayout(NDConfiguration.Layout.ROW_MAJOR)
            b.unsafe.toLayout(NDConfiguration.Layout.ROW_MAJOR)
        then :
            a.matMul(b).toString({it.hasSlimNumbers = true}) == expectedString

        where :
            device  |  expectedString
            'CPU'   |  '(2x4):[24, -8, 8, 0, -1, 28, -14, 7]'
            'GPU'   |  '(2x4):[24, -8, 8, 0, -1, 28, -14, 7]'
    }

}
