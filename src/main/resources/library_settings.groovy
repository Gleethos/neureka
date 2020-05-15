import neureka.Neureka

Neureka.instance {

    settings {

        autoDiff {
            delegate.retainPendingErrorForJITProp = true
            delegate.applyGradientWhenTensorIsUsed = true
        }

        indexing {
            delegate.legacy = false
            delegate.thorough = true
        }

        debug {
            delegate.keepDerivativeTargetPayloads = false
        }

        view {
            delegate.legacy = false
        }
    }

    return "0.1.0"

}