class ComplexNumber(
        private val real: Double, private val imaginary: Double
) {
 
    fun plus( z2 : ComplexNumber ) : ComplexNumber
    {
        return ComplexNumber(this.real + z2.real, this.imaginary + z2.imaginary)
    }

    fun minus( z2 : ComplexNumber ) : ComplexNumber
    {
        return ComplexNumber(this.real - z2.real, this.imaginary - z2.imaginary)
    }

    fun multiply( z2 : ComplexNumber ) : ComplexNumber
    {
        val _real : Double = this.real * z2.real - this.imaginary * z2.imaginary
        val _imaginary : Double = this.real * z2.imaginary + this.imaginary * z2.real
        return ComplexNumber(_real,_imaginary)
    }

    fun divide(z2 : ComplexNumber) : ComplexNumber
    {
        val output : ComplexNumber = multiply(z2.conjugate())
        val div : Double = Math.pow(z2.mod(),2.0)
        return ComplexNumber(output.real/div,output.imaginary/div)
    }

    fun mod() : Double
    {
        return Math.sqrt(Math.pow(this.real,2.0) + Math.pow(this.imaginary,2.0))
    }

    fun pow( power : Int ) : ComplexNumber
    {
        var output = ComplexNumber(this.real,this.imaginary)
        for ( i in 0..power )
        {
            val _real : Double = output.real * this.real - output.imaginary * this.imaginary
            val _imaginary : Double = output.real * this.imaginary + output.imaginary * this.real
            output = ComplexNumber(_real,_imaginary)
        }
        return output
    }

    fun inverse() : ComplexNumber
    {
        return ComplexNumber(1.0,0.0).divide(this)
    }

    fun conjugate() : ComplexNumber
    {
        return ComplexNumber(this.real,-this.imaginary)
    }

    fun square() : ComplexNumber
    {
        val real : Double = this.real * this.real - this.imaginary * this.imaginary
        val imaginary : Double = 2 * this.real * this.imaginary
        return ComplexNumber(real,imaginary)
    }

    override fun toString() : String
    {
        val re = "" + this.real + ""
        val im : String
        if ( this.imaginary < 0 ) im = "" + this.imaginary + "i"
        else im = "+" + this.imaginary + "i"
        return re + im
    }


    override fun equals( z: Any? ): Boolean
    {
        if ( z !is ComplexNumber ) return false
        val a : ComplexNumber =  z
        return (real == a.real) && (imaginary == a.imaginary)
    }

}