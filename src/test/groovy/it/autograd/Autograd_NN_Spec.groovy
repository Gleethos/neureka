package it.autograd

import neureka.Neureka
import neureka.Tsr
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
                graph = node.toString("gv")
            }

        then :
            W1s[0].contains("(2x3):[-1.1843, 0.0146, -1.4647, -1.402, -1.0129, 0.6256]:g:[null]")
            z1s[0].contains("(3x3):[-2.19157E0, -1.00317E0, -0.35091E0, -1.17368E0, -0.55790E0, -0.14060E0, -2.11901E0, -0.66070E0, -1.04761E0]")
            hiddenResults[0].contains("(3x3):[0.10050E0, 0.26831E0, 0.41316E0, 0.23619E0, 0.36403E0, 0.46490E0, 0.10726E0, 0.34058E0, 0.25968E0]; ->d(3x3):[0.09040E0, 0.19632E0, 0.24245E0, 0.18040E0, 0.23151E0, 0.24876E0, 0.09575E0, 0.22458E0, 0.19224E0]")
            W2s[0].contains("(3x1):[1.8095, -0.4269, -1.111]:g:[null]")
            z2s[0].contains("(3x1):[-0.39169E0, -0.24453E0, -0.23981E0]") // This has been checked thoroughly!
            outputResults[0].contains("(3x1):[0.40330E0, 0.43917E0, 0.44033E0]; ->d(3x1):[0.24065E0, 0.24629E0, 0.24643E0]")// This has been checked thoroughly!
            errors[0].contains("(3x1):[0.51669E0, 0.56082E0, 0.44966E0]") // This has been checked thoroughly!
            losses[0].contains("(1x1):[0.26123E0]")

            W1s[1].contains("(2x3):[-1.13651E0, -0.00752E0, -1.52342E0, -1.3438E0, -1.03799E0, 0.55511E0]:g:[null]")
            z1s[1].contains("(3x3):[-2.10151E0, -1.043E0, -0.46055E0, -1.12542E0, -0.57921E0, -0.19933E0, -2.03242E0, -0.69955E0, -1.15333E0]")
            hiddenResults[1].contains("(3x3):[0.10894E0, 0.26057E0, 0.38685E0, 0.24500E0, 0.35911E0, 0.45033E0, 0.11584E0, 0.33191E0, 0.23988E0]; ->d(3x3):[0.09707E0, 0.19267E0, 0.23719E0, 0.18497E0, 0.23015E0, 0.24753E0, 0.10242E0, 0.22174E0, 0.18233E0]")
            W2s[1].contains("(3x1):[1.86651E0, -0.30550E0, -0.96663E0]:g:[null]")
            z2s[1].contains("(3x1):[-0.25019E0, -0.08770E0, -0.11706E0]") // This has been checked thoroughly!
            outputResults[1].contains("(3x1):[0.43777E0, 0.47808E0, 0.47076E0]; ->d(3x1):[0.24612E0, 0.24951E0, 0.24914E0]")// This has been checked thoroughly!
            errors[1].contains("(3x1):[0.48222E0, 0.52191E0, 0.41923E0]") // This has been checked thoroughly!
            losses[1].contains("(1x1):[0.22689E0]")

            losses[2].contains("(1x1):[0.19843E0]")
            losses[3].contains("(1x1):[0.17438E0]")
            losses[4].contains("(1x1):[0.15367E0]")
            losses[5].contains("(1x1):[0.13556E0]")

        and :
            graph.contains("""
]
]    0»1» GraphNode[ sig(I[0]) => (3x1):[0.54268E0, 0.60176E0, 0.56483E0], type='BRANCH'] 
]       \\
]        0»2» GraphNode[ (I[0] @ I[1]) => (3x1):[0.17116E0, 0.41280E0, 0.26080E0], type='BRANCH'] 
]           \\
]            0»1» GraphNode[ sig(I[0]) => (3x3):[0.15178E0, 0.25131E0, 0.32789E0, ... + 6 more], type='BRANCH'] 
]            |  \\
]            |   0»2» GraphNode[ (I[0] @ I[1]) => (3x3):[-1.72064E0, -1.09161E0, -0.71770E0, ... + 6 more], type='BRANCH'] 
]            |      \\
]            |       0»0» GraphNode[ (3x2):[0.6667, 1.0, 0.3333, ... + 3 more], type='LEAVE'] 
]            |       |
]            |       1»0» GraphNode[ (2x3):[-0.88023E0, -0.03096E0, -1.67769E0, ... + 3 more], type='LEAVE RQS GRADIENT'] 
]            |
]            1»0» GraphNode[ (3x1):[2.14504E0, 0.17962E0, -0.43376E0], type='LEAVE RQS GRADIENT'] 
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
                    graph = node.toString("gv")
                }

            then :
                W1s[0].contains("(2x3):[-1.1843, 0.0146, -1.4647, -1.402, -1.0129, 0.6256]:g:[null]")
                z1s[0].contains("(3x1x3):[-2.19157E0, -1.00317E0, -0.35091E0, -1.17368E0, -0.55790E0, -0.14060E0, -2.11901E0, -0.66070E0, -1.04761E0]")
                hiddenResults[0].contains("(3x1x3):[0.10050E0, 0.26831E0, 0.41316E0, 0.23619E0, 0.36403E0, 0.46490E0, 0.10726E0, 0.34058E0, 0.25968E0]; ->d(3x1x3):[0.09040E0, 0.19632E0, 0.24245E0, 0.18040E0, 0.23151E0, 0.24876E0, 0.09575E0, 0.22458E0, 0.19224E0]")
                W2s[0].contains("(3x1):[1.8095, -0.4269, -1.111]:g:[null]")
                z2s[0].contains("(3):[-0.39169E0, -0.24453E0, -0.23981E0]") // This has been checked thoroughly!
                outputResults[0].contains("(3):[0.40330E0, 0.43917E0, 0.44033E0]; ->d(3):[0.24065E0, 0.24629E0, 0.24643E0]")// This has been checked thoroughly!
                errors[0].contains("(3x1):[0.51669E0, 0.56082E0, 0.44966E0]") // This has been checked thoroughly!
                losses[0].contains("(1x1):[0.26123E0]")

                W1s[1].contains("(2x3):[-1.13651E0, -0.00752E0, -1.52342E0, -1.3438E0, -1.03799E0, 0.55511E0]:g:[null]")
                z1s[1].contains("(3x1x3):[-2.10151E0, -1.043E0, -0.46055E0, -1.12542E0, -0.57921E0, -0.19933E0, -2.03242E0, -0.69955E0, -1.15333E0]")
                hiddenResults[1].contains("(3x1x3):[0.10894E0, 0.26057E0, 0.38685E0, 0.24500E0, 0.35911E0, 0.45033E0, 0.11584E0, 0.33191E0, 0.23988E0]; ->d(3x1x3):[0.09707E0, 0.19267E0, 0.23719E0, 0.18497E0, 0.23015E0, 0.24753E0, 0.10242E0, 0.22174E0, 0.18233E0]")
                W2s[1].contains("(3x1):[1.86651E0, -0.30550E0, -0.96663E0]:g:[null]")
                z2s[1].contains("(3):[-0.25019E0, -0.08770E0, -0.11706E0]") // This has been checked thoroughly!
                outputResults[1].contains("(3):[0.43777E0, 0.47808E0, 0.47076E0]; ->d(3):[0.24612E0, 0.24951E0, 0.24914E0]")// This has been checked thoroughly!
                errors[1].contains("(3x1):[0.48222E0, 0.52191E0, 0.41923E0]") // This has been checked thoroughly!
                losses[1].contains("(1x1):[0.22689E0]")

                losses[2].contains("(1x1):[0.19843E0]")
                losses[3].contains("(1x1):[0.17438E0]")
                losses[4].contains("(1x1):[0.15367E0]")
                losses[5].contains("(1x1):[0.13556E0]")

            and :
                graph.contains("""
]
]    0»1» GraphNode[ sig(I[0]) => (3):[0.54268E0, 0.60176E0, 0.56483E0], type='BRANCH'] 
]       \\
]        0»1» GraphNode[ dimtrim(I[0]) => (3):[0.17116E0, 0.41280E0, 0.26080E0], type='BRANCH'] 
]           \\
]            0»2» GraphNode[ (I[0] x I[1]) => (3x1x1x1):[0.17116E0, 0.41280E0, 0.26080E0], type='BRANCH'] 
]               \\
]                0»1» GraphNode[ ([0,1,2,-1]:(I[0])) => (3x1x3x1):[0.15178E0, 0.25131E0, 0.32789E0, ... + 6 more], type='BRANCH'] 
]                |  \\
]                |   0»1» GraphNode[ sig(I[0]) => (3x1x3):[0.15178E0, 0.25131E0, 0.32789E0, ... + 6 more], type='BRANCH'] 
]                |      \\
]                |       0»1» GraphNode[ dimtrim(I[0]) => (3x1x3):[-1.72064E0, -1.09161E0, -0.71770E0, ... + 6 more], type='BRANCH'] 
]                |          \\
]                |           0»2» GraphNode[ (I[0] x I[1]) => (3x1x3):[-1.72064E0, -1.09161E0, -0.71770E0, ... + 6 more], type='BRANCH'] 
]                |              \\
]                |               0»1» GraphNode[ ([0,1,-1]:(I[0])) => (3x2x1):[0.6667, 1.0, 0.3333, ... + 3 more], type='BRANCH'] 
]                |               |  \\
]                |               |   0»0» GraphNode[ (3x2):[0.6667, 1.0, 0.3333, ... + 3 more], type='LEAVE'] 
]                |               |
]                |               1»1» GraphNode[ ([-1,0,1]:(I[0])) => (1x2x3):[-0.88023E0, -0.03096E0, -1.67769E0, ... + 3 more], type='BRANCH'] 
]                |                  \\
]                |                   0»0» GraphNode[ (2x3):[-0.88023E0, -0.03096E0, -1.67769E0, ... + 3 more], type='LEAVE RQS GRADIENT'] 
]                |
]                1»1» GraphNode[ ([-1,-1,0,1]:(I[0])) => (1x1x3x1):[2.14504E0, 0.17962E0, -0.43376E0], type='BRANCH'] 
]                   \\
]                    0»0» GraphNode[ (3x1):[2.14504E0, 0.17962E0, -0.43376E0], type='LEAVE RQS GRADIENT'] 
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
                graph = node.toString("gv")
            }

        then :
            W1s[0].contains("(2x3):[-1.1843E0, 0.01460E0, -1.4647E0, -1.402E0, -1.0129E0, 0.62559E0]:g:[null]")
            z1s[0].contains("(3x1x3):[-2.19157E0, -1.00317E0, -0.35091E0, -1.17368E0, -0.55790E0, -0.14060E0, -2.11901E0, -0.66070E0, -1.04761E0]")
            hiddenResults[0].contains("(3x1x3):[0.10050E0, 0.26831E0, 0.41316E0, 0.23619E0, 0.36403E0, 0.46490E0, 0.10726E0, 0.34058E0, 0.25968E0]; ->d(3x1x3):[0.09040E0, 0.19632E0, 0.24245E0, 0.18040E0, 0.23151E0, 0.24876E0, 0.09575E0, 0.22458E0, 0.19224E0]")
            W2s[0].contains("(3x1):[1.8095, -0.4269, -1.111]:g:[null]")
            z2s[0].contains("(3):[-0.39169E0, -0.24453E0, -0.23981E0]") // This has been checked thoroughly!
            outputResults[0].contains("(3):[0.40330E0, 0.43917E0, 0.44033E0]; ->d(3):[0.24065E0, 0.24629E0, 0.24643E0]")// This has been checked thoroughly!
            errors[0].contains("(3x1):[0.51669E0, 0.56082E0, 0.44966E0]") // This has been checked thoroughly!
            losses[0].contains("(1x1):[0.26123E0]")

            W1s[1].contains("(2x3):[-1.13651E0, -0.00752E0, -1.52342E0, -1.3438E0, -1.03799E0, 0.55511E0]:g:[null]")
            z1s[1].contains("(3x1x3):[-2.10151E0, -1.043E0, -0.46055E0, -1.12542E0, -0.57921E0, -0.19933E0, -2.03242E0, -0.69955E0, -1.15333E0]")
            hiddenResults[1].contains("(3x1x3):[0.10894E0, 0.26057E0, 0.38685E0, 0.24500E0, 0.35911E0, 0.45033E0, 0.11584E0, 0.33191E0, 0.23988E0]; ->d(3x1x3):[0.09707E0, 0.19267E0, 0.23719E0, 0.18497E0, 0.23015E0, 0.24753E0, 0.10242E0, 0.22174E0, 0.18233E0]")
            W2s[1].contains("(3x1):[1.86651E0, -0.30550E0, -0.96663E0]:g:[null]")
            z2s[1].contains("(3):[-0.25019E0, -0.08770E0, -0.11706E0]") // This has been checked thoroughly!
            outputResults[1].contains("(3):[0.43777E0, 0.47808E0, 0.47076E0]; ->d(3):[0.24612E0, 0.24951E0, 0.24914E0]")// This has been checked thoroughly!
            errors[1].contains("(3x1):[0.48222E0, 0.52191E0, 0.41923E0]") // This has been checked thoroughly!
            losses[1].contains("(1x1):[0.22689E0]")

            losses[2].contains("(1x1):[0.19843E0]")
            losses[3].contains("(1x1):[0.17438E0]")
            losses[4].contains("(1x1):[0.15367E0]")
            losses[5].contains("(1x1):[0.13556E0]")

        and :
            graph.contains("""
]
]    0»1» GraphNode[ sig(I[0]) => (3):[0.54268E0, 0.60176E0, 0.56483E0], type='BRANCH'] 
]       \\
]        0»1» GraphNode[ dimtrim(I[0]) => (3):[0.17116E0, 0.41280E0, 0.26080E0], type='BRANCH'] 
]           \\
]            0»2» GraphNode[ (I[0] x I[1]) => (3x1x1x1):[0.17116E0, 0.41280E0, 0.26080E0], type='BRANCH'] 
]               \\
]                0»1» GraphNode[ ([0,1,2,-1]:(I[0])) => (3x1x3x1):[0.15178E0, 0.25131E0, 0.32789E0, ... + 6 more], type='BRANCH'] 
]                |  \\
]                |   0»1» GraphNode[ sig(I[0]) => (3x1x3):[0.15178E0, 0.25131E0, 0.32789E0, ... + 6 more], type='BRANCH'] 
]                |      \\
]                |       0»1» GraphNode[ dimtrim(I[0]) => (3x1x3):[-1.72064E0, -1.09161E0, -0.71770E0, ... + 6 more], type='BRANCH'] 
]                |          \\
]                |           0»2» GraphNode[ (I[0] x I[1]) => (3x1x3):[-1.72064E0, -1.09161E0, -0.71770E0, ... + 6 more], type='BRANCH'] 
]                |              \\
]                |               0»1» GraphNode[ ([0,1,-1]:(I[0])) => (3x2x1):[0.66670E0, 1.0, 0.33329E0, ... + 3 more], type='BRANCH'] 
]                |               |  \\
]                |               |   0»0» GraphNode[ (3x2):[0.66670E0, 1.0, 0.33329E0, ... + 3 more], type='LEAVE'] 
]                |               |
]                |               1»1» GraphNode[ ([-1,0,1]:(I[0])) => (1x2x3):[-0.88023E0, -0.03096E0, -1.67769E0, ... + 3 more], type='BRANCH'] 
]                |                  \\
]                |                   0»0» GraphNode[ (2x3):[-0.88023E0, -0.03096E0, -1.67769E0, ... + 3 more], type='LEAVE RQS GRADIENT'] 
]                |
]                1»1» GraphNode[ ([-1,-1,0,1]:(I[0])) => (1x1x3x1):[2.14504E0, 0.17962E0, -0.43376E0], type='BRANCH'] 
]                   \\
]                    0»0» GraphNode[ (3x1):[2.14504E0, 0.17962E0, -0.43376E0], type='LEAVE RQS GRADIENT'] 
]
""")

    }



    def 'Autograd work for simple matrix multiplications.'() {

        given :
            def a = Tsr.of([2, 3], -1d..4d).setRqsGradient(true)
            def b = Tsr.of([3, 1], [-4d, -2d, 0d]).setRqsGradient(true)

        when :
            def c = a.matMul(b)

        then :
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
            c.backward(Tsr.of(c.shape(), [-1d, 1d])) // (2x1):[-1, 1]

        then :
            a.toString() == "(2x3):[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:g:[4.0, 2.0, 0.0, -4.0, -2.0, 0.0]"
            b.toString() == "(3x1):[-4.0, -2.0, 0.0]:g:[3.0, 3.0, 3.0]"
    }


    @IgnoreIf({!(Neureka.get().canAccessOpenCL() || !(device instanceof OpenCLDevice))})
    def 'Autograd works for 2 matrix multiplications in a row.'(Device<?> device) {

        given :
            def a = Tsr.of([2, 3], -1d..4d).setRqsGradient(true).to(device)
            def b = Tsr.of([3, 1], [-4d, -2d, 0d]).setRqsGradient(true).to(device)
            def x = Tsr.of([[0.5d, 0.5d]]).to(device)

        when :
            def c = a.matMul(b)
            def o = x.matMul(c)

        then :
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
        and :
            x.toString() == "(1x2):[0.5, 0.5]"
            o.toString() == "(1x1):[-5.0]; ->d(2x1):[0.5, 0.5]"

        when :
            o.backward()

        then :
            a.toString() == "(2x3):[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:g:[-2.0, -1.0, 0.0, -2.0, -1.0, 0.0]"
            b.toString() == "(3x1):[-4.0, -2.0, 0.0]:g:[0.5, 1.5, 2.5]"

        where :
            device << [CPU.get(), Device.find('first gpu')]

    }


}
