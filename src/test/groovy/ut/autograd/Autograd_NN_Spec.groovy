package ut.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import neureka.calculus.Function
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import neureka.view.TsrStringSettings
import spock.lang.IgnoreIf
import spock.lang.Specification

class Autograd_NN_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2>Simple Neural Network autograd integration test.</h2>
            <p>
                The integration test below has been implemented by using
                the following code and the result it produces as reference : <br>
                https://medium.com/dair-ai/a-simple-neural-network-from-scratch-with-pytorch-and-google-colab-c7f3830618e0 <br>
                <br>
                The following seed has been used to assure reproducibility : <br>
                'torch.manual_seed(503672689411)'
            </p>
            """
    }

    def setup() {
        Neureka.get().reset()
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


    def 'Autograd works in a simple mat-mul based feed forward neural network.'()
    {
        given :
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested( false )
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed( false )
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp( false )
            def X = Tsr.of(
                    [[0.6667, 1.0000],
                     [0.3333, 0.5556],
                     [1.0000, 0.6667]]
            )
            def y = Tsr.of(
                    [[0.9200],
                     [1.0000],
                     [0.8900]]
            )
            def sig = Function.of("sig(I[0])")
            def W1 = Tsr.of(
                    [[-1.1843,  0.0146, -1.4647],
                     [-1.4020, -1.0129,  0.6256]]
            ).setRqsGradient(true)
            def W2 = Tsr.of(
                    [[ 1.8095],
                     [-0.4269],
                     [-1.1110]]
            ).setRqsGradient(true)

            def W1s = []
            def z1s = []
            def hiddenResults = []
            def W2s = []
            def z2s = []
            def outputResults = []
            def errors = []
            def losses = []

            def forwardAndBackward = ( Tsr x ) ->
            {
                W1s.add(W1.toString())
                def z1 = x.matMul(W1)
                z1s.add(z1.toString())
                def hidden = sig(z1)
                hiddenResults.add(hidden.toString())
                W2s.add(W2.toString())
                def z2 = hidden.matMul(W2)
                z2s.add(z2.toString())
                def pred = sig(z2)
                outputResults.add(pred.toString())
                def error = (y - pred)
                errors.add(error.toString())
                def loss = (error**2).mean()
                losses.add(loss.toString())
                pred.backward(error)
                W1.applyGradient()
                W2.applyGradient()
                return pred
            }

        when :
            def graph
            6.times {
                def node = forwardAndBackward(X).graphNode
                graph = node.toString(GraphNode.Print.FANCY)
            }

        then :
            W1s[0].contains("(2x3):[-1.1843, 0.0146, -1.4647, -1.402, -1.0129, 0.6256]:g:[null]")
            z1s[0].contains("(3x3):[-2.19157, -1.00317, -0.35091, -1.17368, -0.55790, -0.14060, -2.11901, -0.66070, -1.04761]")
            hiddenResults[0].contains("(3x3):[0.10050, 0.26831, 0.41316, 0.23619, 0.36403, 0.46490, 0.10726, 0.34058, 0.25968]; ->d(3x3):[0.09040, 0.19632, 0.24245, 0.18040, 0.23151, 0.24876, 0.09575, 0.22458, 0.19224]")
            W2s[0].contains("(3x1):[1.8095, -0.4269, -1.111]:g:[null]")
            z2s[0].contains("(3x1):[-0.39169, -0.24453, -0.23981]") // This has been checked thoroughly!
            outputResults[0].contains("(3x1):[0.40330, 0.43917, 0.44033]; ->d(3x1):[0.24065, 0.24629, 0.24643]")// This has been checked thoroughly!
            errors[0].contains("(3x1):[0.51669, 0.56082, 0.44966]") // This has been checked thoroughly!
            losses[0].contains("(1x1):[0.26123]")

            W1s[1].contains("(2x3):[-1.13651, -0.00752, -1.52342, -1.3438, -1.03799, 0.55511]:g:[null]")
            z1s[1].contains("(3x3):[-2.10151, -1.043, -0.46055, -1.12542, -0.57921, -0.19933, -2.03242, -0.69955, -1.15333]")
            hiddenResults[1].contains("(3x3):[0.10894, 0.26057, 0.38685, 0.24500, 0.35911, 0.45033, 0.11584, 0.33191, 0.23988]; ->d(3x3):[0.09707, 0.19267, 0.23719, 0.18497, 0.23015, 0.24753, 0.10242, 0.22174, 0.18233]")
            W2s[1].contains("(3x1):[1.86651, -0.30550, -0.96663]:g:[null]")
            z2s[1].contains("(3x1):[-0.25019, -0.08770, -0.11706]") // This has been checked thoroughly!
            outputResults[1].contains("(3x1):[0.43777, 0.47808, 0.47076]; ->d(3x1):[0.24612, 0.24951, 0.24914]")// This has been checked thoroughly!
            errors[1].contains("(3x1):[0.48222, 0.52191, 0.41923]") // This has been checked thoroughly!
            losses[1].contains("(1x1):[0.22689]")

            losses[2].contains("(1x1):[0.19843]")
            losses[3].contains("(1x1):[0.17438]")
            losses[4].contains("(1x1):[0.15367]")
            losses[5].contains("(1x1):[0.13556]")

        and :
            graph.contains("""
]
]    0»1» GraphNode[ sig(I[0]) => (3x1):[0.54268, 0.60176, 0.56483], type='BRANCH'] 
]       \\
]        0»2» GraphNode[ (I[0] @ I[1]) => (3x1):[0.17116, 0.41280, 0.26080], type='BRANCH'] 
]           \\
]            0»1» GraphNode[ sig(I[0]) => (3x3):[0.15178, 0.25131, 0.32789, ... + 6 more], type='BRANCH'] 
]            |  \\
]            |   0»2» GraphNode[ (I[0] @ I[1]) => (3x3):[-1.72064, -1.09161, -0.71770, ... + 6 more], type='BRANCH'] 
]            |      \\
]            |       0»0» GraphNode[ (3x2):[0.6667, 1.0, 0.3333, ... + 3 more], type='LEAVE'] 
]            |       |
]            |       1»0» GraphNode[ (2x3):[-0.88023, -0.03096, -1.67769, ... + 3 more], type='LEAVE RQS GRADIENT'] 
]            |
]            1»0» GraphNode[ (3x1):[2.14504, 0.17962, -0.43376], type='LEAVE RQS GRADIENT'] 
]
""")

    }


    def 'Autograd works in a simple convolutional dot product based feed forward neural network.'() {
        given :
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested( false )
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed( false )
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp( false )
            var X = Tsr.of(
                                [[0.6667, 1.0000],
                                 [0.3333, 0.5556],
                                 [1.0000, 0.6667]]
                            )

            var y = Tsr.of(
                                [[0.9200],
                                 [1.0000],
                                 [0.8900]]
                            )

            var sig = Function.of("sig(I[0])")

            var W1 = Tsr.of(
                                [[-1.1843,  0.0146, -1.4647],
                                 [-1.4020, -1.0129,  0.6256]]
                            )
                            .setRqsGradient(true)

            var W2 = Tsr.of(
                                [[ 1.8095],
                                 [-0.4269],
                                 [-1.1110]]
                            )
                            .setRqsGradient(true)

            def W1s = []
            def z1s = []
            def hiddenResults = []
            def W2s = []
            def z2s = []
            def outputResults = []
            def errors = []
            def losses = []

            def forwardAndBackward = ( Tsr x ) ->
            {
                W1s.add(W1.toString())
                def z1 = x.convDot(W1)
                z1s.add(z1.toString())
                def hidden = sig(z1)
                hiddenResults.add(hidden.toString())
                W2s.add(W2.toString())
                def z2 = hidden.convDot(W2)
                z2s.add(z2.toString())
                def pred = sig(z2)
                outputResults.add(pred.toString())
                def error = (y - pred)
                errors.add(error.toString())
                def loss = (error**2).mean()
                losses.add(loss.toString())
                pred.backward(error)
                W1.applyGradient()
                W2.applyGradient()
                return pred
            }

            when :
                def graph
                6.times {
                    def node = forwardAndBackward(X).graphNode
                    graph = node.toString(GraphNode.Print.FANCY)
                }

            then :
                W1s[0].contains("(2x3):[-1.1843, 0.0146, -1.4647, -1.402, -1.0129, 0.6256]:g:[null]")
                z1s[0].contains("(3x1x3):[-2.19157, -1.00317, -0.35091, -1.17368, -0.55790, -0.14060, -2.11901, -0.66070, -1.04761]")
                hiddenResults[0].contains("(3x1x3):[0.10050, 0.26831, 0.41316, 0.23619, 0.36403, 0.46490, 0.10726, 0.34058, 0.25968]; ->d(3x1x3):[0.09040, 0.19632, 0.24245, 0.18040, 0.23151, 0.24876, 0.09575, 0.22458, 0.19224]")
                W2s[0].contains("(3x1):[1.8095, -0.4269, -1.111]:g:[null]")
                z2s[0].contains("(3):[-0.39169, -0.24453, -0.23981]") // This has been checked thoroughly!
                outputResults[0].contains("(3):[0.40330, 0.43917, 0.44033]; ->d(3):[0.24065, 0.24629, 0.24643]")// This has been checked thoroughly!
                errors[0].contains("(3x1):[0.51669, 0.56082, 0.44966]") // This has been checked thoroughly!
                losses[0].contains("(1x1):[0.26123]")

                W1s[1].contains("(2x3):[-1.13651, -0.00752, -1.52342, -1.3438, -1.03799, 0.55511]:g:[null]")
                z1s[1].contains("(3x1x3):[-2.10151, -1.043, -0.46055, -1.12542, -0.57921, -0.19933, -2.03242, -0.69955, -1.15333]")
                hiddenResults[1].contains("(3x1x3):[0.10894, 0.26057, 0.38685, 0.24500, 0.35911, 0.45033, 0.11584, 0.33191, 0.23988]; ->d(3x1x3):[0.09707, 0.19267, 0.23719, 0.18497, 0.23015, 0.24753, 0.10242, 0.22174, 0.18233]")
                W2s[1].contains("(3x1):[1.86651, -0.30550, -0.96663]:g:[null]")
                z2s[1].contains("(3):[-0.25019, -0.08770, -0.11706]") // This has been checked thoroughly!
                outputResults[1].contains("(3):[0.43777, 0.47808, 0.47076]; ->d(3):[0.24612, 0.24951, 0.24914]")// This has been checked thoroughly!
                errors[1].contains("(3x1):[0.48222, 0.52191, 0.41923]") // This has been checked thoroughly!
                losses[1].contains("(1x1):[0.22689]")

                losses[2].contains("(1x1):[0.19843]")
                losses[3].contains("(1x1):[0.17438]")
                losses[4].contains("(1x1):[0.15367]")
                losses[5].contains("(1x1):[0.13556]")

            and :
                graph.contains("""
]
]    0»1» GraphNode[ sig(I[0]) => (3):[0.54268, 0.60176, 0.56483], type='BRANCH'] 
]       \\
]        0»1» GraphNode[ dimtrim(I[0]) => (3):[0.17116, 0.41280, 0.26080], type='BRANCH'] 
]           \\
]            0»2» GraphNode[ (I[0] x I[1]) => (3x1x1x1):[0.17116, 0.41280, 0.26080], type='BRANCH'] 
]               \\
]                0»1» GraphNode[ ([0,1,2,-1]:(I[0])) => (3x1x3x1):[0.15178, 0.25131, 0.32789, ... + 6 more], type='BRANCH'] 
]                |  \\
]                |   0»1» GraphNode[ sig(I[0]) => (3x1x3):[0.15178, 0.25131, 0.32789, ... + 6 more], type='BRANCH'] 
]                |      \\
]                |       0»1» GraphNode[ dimtrim(I[0]) => (3x1x3):[-1.72064, -1.09161, -0.71770, ... + 6 more], type='BRANCH'] 
]                |          \\
]                |           0»2» GraphNode[ (I[0] x I[1]) => (3x1x3):[-1.72064, -1.09161, -0.71770, ... + 6 more], type='BRANCH'] 
]                |              \\
]                |               0»1» GraphNode[ ([0,1,-1]:(I[0])) => (3x2x1):[0.6667, 1.0, 0.3333, ... + 3 more], type='BRANCH'] 
]                |               |  \\
]                |               |   0»0» GraphNode[ (3x2):[0.6667, 1.0, 0.3333, ... + 3 more], type='LEAVE'] 
]                |               |
]                |               1»1» GraphNode[ ([-1,0,1]:(I[0])) => (1x2x3):[-0.88023, -0.03096, -1.67769, ... + 3 more], type='BRANCH'] 
]                |                  \\
]                |                   0»0» GraphNode[ (2x3):[-0.88023, -0.03096, -1.67769, ... + 3 more], type='LEAVE RQS GRADIENT'] 
]                |
]                1»1» GraphNode[ ([-1,-1,0,1]:(I[0])) => (1x1x3x1):[2.14504, 0.17962, -0.43376], type='BRANCH'] 
]                   \\
]                    0»0» GraphNode[ (3x1):[2.14504, 0.17962, -0.43376], type='LEAVE RQS GRADIENT'] 
]
""")

    }


    def 'Autograd works in a simple convolutional dot product and float based feed forward neural network.'() {
        given :
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested( false )
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed( false )
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp( false )
            var X = Tsr.of(
                                    [[0.6667f, 1.0000f],
                                     [0.3333f, 0.5556f],
                                     [1.0000f, 0.6667f]]
                            )

            var y = Tsr.of(
                                    [[0.9200f],
                                     [1.0000f],
                                     [0.8900f]]
                            )

            var sig = Function.of("sig(I[0])")

            var W1 = Tsr.of(
                            [[-1.1843f,  0.0146f, -1.4647f],
                             [-1.4020f, -1.0129f,  0.6256f]]
                    )
                    .setRqsGradient(true)

            var W2 = Tsr.of(
                                [[ 1.8095],
                                 [-0.4269],
                                 [-1.1110]]
                        )
                        .setRqsGradient(true)

            def W1s = []
            def z1s = []
            def hiddenResults = []
            def W2s = []
            def z2s = []
            def outputResults = []
            def errors = []
            def losses = []

            def forwardAndBackward = ( Tsr x ) ->
            {
                W1s.add(W1.toString())
                def z1 = x.convDot(W1)
                z1s.add(z1.toString())
                def hidden = sig(z1)
                hiddenResults.add(hidden.toString())
                W2s.add(W2.toString())
                def z2 = hidden.convDot(W2)
                z2s.add(z2.toString())
                def pred = sig(z2)
                outputResults.add(pred.toString())
                def error = (y - pred)
                errors.add(error.toString())
                def loss = (error**2).mean()
                losses.add(loss.toString())
                pred.backward(error)
                W1.applyGradient()
                W2.applyGradient()
                return pred
            }

        when :
            def graph
            6.times {
                def node = forwardAndBackward(X).graphNode
                graph = node.toString(GraphNode.Print.FANCY)
            }

        then :
            W1s[0].contains("(2x3):[-1.1843, 0.01460, -1.4647, -1.402, -1.0129, 0.62559]:g:[null]")
            z1s[0].contains("(3x1x3):[-2.19157, -1.00317, -0.35091, -1.17368, -0.55790, -0.14060, -2.11901, -0.66070, -1.04761]")
            hiddenResults[0].contains("(3x1x3):[0.10050, 0.26831, 0.41316, 0.23619, 0.36403, 0.46490, 0.10726, 0.34058, 0.25968]; ->d(3x1x3):[0.09040, 0.19632, 0.24245, 0.18040, 0.23151, 0.24876, 0.09575, 0.22458, 0.19224]")
            W2s[0].contains("(3x1):[1.8095, -0.4269, -1.111]:g:[null]")
            z2s[0].contains("(3):[-0.39169, -0.24453, -0.23981]") // This has been checked thoroughly!
            outputResults[0].contains("(3):[0.40330, 0.43917, 0.44033]; ->d(3):[0.24065, 0.24629, 0.24643]")// This has been checked thoroughly!
            errors[0].contains("(3x1):[0.51669, 0.56082, 0.44966]") // This has been checked thoroughly!
            losses[0].contains("(1x1):[0.26123]")

            W1s[1].contains("(2x3):[-1.13651, -0.00752, -1.52342, -1.3438, -1.03799, 0.55511]:g:[null]")
            z1s[1].contains("(3x1x3):[-2.10151, -1.043, -0.46055, -1.12542, -0.57921, -0.19933, -2.03242, -0.69955, -1.15333]")
            hiddenResults[1].contains("(3x1x3):[0.10894, 0.26057, 0.38685, 0.24500, 0.35911, 0.45033, 0.11584, 0.33191, 0.23988]; ->d(3x1x3):[0.09707, 0.19267, 0.23719, 0.18497, 0.23015, 0.24753, 0.10242, 0.22174, 0.18233]")
            W2s[1].contains("(3x1):[1.86651, -0.30550, -0.96663]:g:[null]")
            z2s[1].contains("(3):[-0.25019, -0.08770, -0.11706]") // This has been checked thoroughly!
            outputResults[1].contains("(3):[0.43777, 0.47808, 0.47076]; ->d(3):[0.24612, 0.24951, 0.24914]")// This has been checked thoroughly!
            errors[1].contains("(3x1):[0.48222, 0.52191, 0.41923]") // This has been checked thoroughly!
            losses[1].contains("(1x1):[0.22689]")

            losses[2].contains("(1x1):[0.19843]")
            losses[3].contains("(1x1):[0.17438]")
            losses[4].contains("(1x1):[0.15367]")
            losses[5].contains("(1x1):[0.13556]")

        and :
            graph.contains("""
]
]    0»1» GraphNode[ sig(I[0]) => (3):[0.54268, 0.60176, 0.56483], type='BRANCH'] 
]       \\
]        0»1» GraphNode[ dimtrim(I[0]) => (3):[0.17116, 0.41280, 0.26080], type='BRANCH'] 
]           \\
]            0»2» GraphNode[ (I[0] x I[1]) => (3x1x1x1):[0.17116, 0.41280, 0.26080], type='BRANCH'] 
]               \\
]                0»1» GraphNode[ ([0,1,2,-1]:(I[0])) => (3x1x3x1):[0.15178, 0.25131, 0.32789, ... + 6 more], type='BRANCH'] 
]                |  \\
]                |   0»1» GraphNode[ sig(I[0]) => (3x1x3):[0.15178, 0.25131, 0.32789, ... + 6 more], type='BRANCH'] 
]                |      \\
]                |       0»1» GraphNode[ dimtrim(I[0]) => (3x1x3):[-1.72064, -1.09161, -0.71770, ... + 6 more], type='BRANCH'] 
]                |          \\
]                |           0»2» GraphNode[ (I[0] x I[1]) => (3x1x3):[-1.72064, -1.09161, -0.71770, ... + 6 more], type='BRANCH'] 
]                |              \\
]                |               0»1» GraphNode[ ([0,1,-1]:(I[0])) => (3x2x1):[0.66670, 1.0, 0.33329, ... + 3 more], type='BRANCH'] 
]                |               |  \\
]                |               |   0»0» GraphNode[ (3x2):[0.66670, 1.0, 0.33329, ... + 3 more], type='LEAVE'] 
]                |               |
]                |               1»1» GraphNode[ ([-1,0,1]:(I[0])) => (1x2x3):[-0.88023, -0.03096, -1.67769, ... + 3 more], type='BRANCH'] 
]                |                  \\
]                |                   0»0» GraphNode[ (2x3):[-0.88023, -0.03096, -1.67769, ... + 3 more], type='LEAVE RQS GRADIENT'] 
]                |
]                1»1» GraphNode[ ([-1,-1,0,1]:(I[0])) => (1x1x3x1):[2.14504, 0.17962, -0.43376], type='BRANCH'] 
]                   \\
]                    0»0» GraphNode[ (3x1):[2.14504, 0.17962, -0.43376], type='LEAVE RQS GRADIENT'] 
]
""")

    }



    def 'Autograd work for simple matrix multiplications.'( Class<?> type ) {

        given :
            var a = Tsr.of([2, 3], -1f..4f).setRqsGradient(true).unsafe.toType(type)
            var b = Tsr.of([3, 1], [-4d, -2d, 0d]).setRqsGradient(true).unsafe.toType(type)

        when :
            var c = a.matMul(b)
        then :
            c.valueClass == type
        and :
            a.toString() == "(2x3):[" +
                                "-1.0, 0.0, 1.0, " +
                                "2.0, 3.0, 4.0" +
                            "]:g:[null]"
        and :
            b.toString() == "(3x1):[" +
                                    "-4.0, " +
                                    "-2.0, " +
                                    "0.0" +
                                "]:g:[null]"
        and :
            def cStr = c.toString()
            cStr.contains "(2x1):[4.0, -14.0]"
            cStr.contains "->d(3x2):[-1.0, 2.0, 0.0, 3.0, 1.0, 4.0]"
            cStr.contains "->d(1x3):[-4.0, -2.0, 0.0]"

        when :
            c.backward(Tsr.of(c.shape, [-1d, 1d]).unsafe.toType(type)) // (2x1):[-1, 1]

        then :
            a.toString() == "(2x3):[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:g:[4.0, 2.0, 0.0, -4.0, -2.0, 0.0]"
            b.toString() == "(3x1):[-4.0, -2.0, 0.0]:g:[3.0, 3.0, 3.0]"

        where :
            type << [Double, Float]

    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null })
    def 'Autograd works for 2 matrix multiplications in a row.'( Device<?> device ) 
    {
        given : """
            We create 3 tensors with the shapes (2x3), (3x1) and (2x1) for matrix multiplication.
            All of them ought to be stored on the provided device and 
            only the first 2 require gradients, whereas the third one does not.
            We use these tensors to mimic 2 linear forward passes in a neural network.
        """
            def a = Tsr.of([2, 3], -1d..4d).setRqsGradient(true).to(device)
            def b = Tsr.of([3, 1], [-4d, -2d, 0d]).setRqsGradient(true).to(device)
            def x = Tsr.of([[0.5d, 0.5d]]).to(device)

        expect : 'Initially none of the tensors requiring gradients have any (see "g:[null]").'
            a.toString() == "(2x3):[" +
                    "-1.0, 0.0, 1.0, " +
                    "2.0, 3.0, 4.0" +
                    "]:g:[null]"
            b.toString() == "(3x1):[" +
                    "-4.0, " +
                    "-2.0, " +
                    "0.0" +
                    "]:g:[null]"
            x.toString() == "(1x2):[0.5, 0.5]"

        when : 'We perform 2 matrix multiplications in a row, using all 3 previously created tensors...'
            def c = a.matMul(b)
            def o = x.matMul(c)

        then : 'The results from the two matrix multiplications are as expected.'
            def cStr = c.toString()
            cStr.contains "(2x1):[4.0, -14.0]"
            cStr.contains "->d(3x2):[-1.0, 2.0, 0.0, 3.0, 1.0, 4.0]"
            cStr.contains "->d(1x3):[-4.0, -2.0, 0.0]"
            o.toString() == "(1x1):[-5.0]; ->d(2x1):[0.5, 0.5]"


        and : 'We still expect the first 2 tensors to not yet have any gradients (see "g:[null]").'
            a.toString() == "(2x3):[" +
                                    "-1.0, 0.0, 1.0, " +
                                    "2.0, 3.0, 4.0" +
                                "]:g:[null]"
            b.toString() == "(3x1):[" +
                                    "-4.0, " +
                                    "-2.0, " +
                                    "0.0" +
                                "]:g:[null]"
            x.toString() == "(1x2):[0.5, 0.5]"

        when : 'We perform back-propagation...'
            o.backward()

        then : 'Contrary to before, the 2 matrices now do have the expected gradients automatically generated by the aut-grad system.'
            a.toString() == "(2x3):[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:g:[-2.0, -1.0, 0.0, -2.0, -1.0, 0.0]"
            b.toString() == "(3x1):[-4.0, -2.0, 0.0]:g:[0.5, 1.5, 2.5]"

        where : 'We test this feature on both the CPU as well as the GPU.'
            device << [CPU.get(), Device.get('first gpu')]

    }


}
