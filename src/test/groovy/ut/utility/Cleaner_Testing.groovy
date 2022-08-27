package ut.utility

import neureka.Tsr
import neureka.devices.DeviceCleaner
import spock.lang.Specification
import spock.lang.Title
import spock.lang.Narrative
import testutility.Sleep

@Title("How Neureka Cleans Up")
@Narrative ('''

    Under the hood 
    Neureka deals whith large arrays of
    data, which are often times 
    native data arrays requiring explicit
    memory freeing!
    This freeing of memory can happen at any time
    during the livetime of a nd-array, however
    it should happen at least up until the nd-arra/tensor
    objects representing their referenced data arrays become
    eligible for garbage collection.
    This specification ensures that the custom garbage
    cleaner implementation used by Neureka fulfills this role 

''')
class Cleaner_Testing extends Specification
{

    def 'The DeviceCleaner triggers registersd cleaner actions when things are eligable for GC.'(){

        given :
            def cleaner = DeviceCleaner.getInstance()
            def refCount = 10
            def r1 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r2 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r3 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r4 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r5 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r6 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r7 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r8 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r9 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            def r10 = Tsr.ofDoubles().withShape( 2, 4 ).all(0)

            cleaner.register( r1, {refCount--} )
            cleaner.register( r2, {refCount--} )
            cleaner.register( r3, {refCount--} )
            cleaner.register( r4, {refCount--} )
            cleaner.register( r5, {refCount--} )
            cleaner.register( r6, {refCount--} )
            cleaner.register( r7, {refCount--} )
            cleaner.register( r8, {refCount--} )
            cleaner.register( r9, {refCount--} )
            cleaner.register( r10,{refCount--} )

        when :
            r1 = null
            r3 = null
            System.gc()
            Sleep.until(250, {
                refCount == 8 && cleaner._registered == 8
            })
            System.gc()
            Sleep.until(250, {
                refCount == 8 && cleaner._registered == 8
            })

        then :
            refCount == 8
            cleaner._registered == 8

        when :
            r2 = null
            r4 = null
            System.gc()
            Sleep.until(250, {
                refCount == 6 && cleaner._registered == 6
            })
            System.gc()
            Sleep.until(250, {
                refCount == 6 && cleaner._registered == 6
            })

        then :
            refCount == 6
            cleaner._registered == 6

    }


}
