import neureka.Neureka

Neureka.instance {

    settings {

        autoDiff {
            delegate.retainGraphDerivativesAfterBackward = false
            delegate.retainPendingErrorForJITProp = true
            delegate.applyGradientWhenTensorIsUsed = true
        }

        indexing {
            delegate.legacy = false
        }

        debug {
            delegate.keepDerivativeTargetPayloads = false
        }
    }

}