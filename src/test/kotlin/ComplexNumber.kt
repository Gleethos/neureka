class ComplexNumber(
        private val real: Double, private val imaginary: Double
) {

    /**
     *  @return A new complex number which is the sum of this complex number and the one passed to this function.
     */
    fun plus( z2 : ComplexNumber ) = ComplexNumber(this.real + z2.real, this.imaginary + z2.imaginary)

    /**
     * @return A new complex number holding the subtraction between this value and the value of the provided complex number.
     */
    fun minus( z2 : ComplexNumber ) = ComplexNumber(this.real - z2.real, this.imaginary - z2.imaginary)

    /**
     * @return A new complex number which is the product of this complex number and the one passed to this function.
     */
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

    fun mod() = Math.sqrt(Math.pow(this.real,2.0) + Math.pow(this.imaginary,2.0))

    /**
     * @return A new complex number produced by raising it to the power of the provided integer.
     */
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

    /**
     * @return A new complex number representing the inverse of this number.
     */
    fun inverse() = ComplexNumber(1.0,0.0).divide(this)

    fun conjugate() = ComplexNumber(this.real,-this.imaginary)

    /**
     * @return A new complex number holding the squared value of this complex number.
     */
    fun square() : ComplexNumber
    {
        val real : Double = this.real * this.real - this.imaginary * this.imaginary
        val imaginary : Double = 2 * this.real * this.imaginary
        return ComplexNumber(real,imaginary)
    }

    /**
     * @return A string representation of this complex number.
     */
    override fun toString() : String
    {
        val re = "" + this.real + ""
        val im : String
        if ( this.imaginary < 0 ) im = "" + this.imaginary + "i"
        else im = "+" + this.imaginary + "i"
        return re + im
    }

    /**
     * @return The truth value determining if the value of this complex number is equals to the provided object.
     */
    override fun equals( z: Any? ): Boolean
    {
        if ( z !is ComplexNumber ) return false
        val a : ComplexNumber =  z
        return (real == a.real) && (imaginary == a.imaginary)
    }

}