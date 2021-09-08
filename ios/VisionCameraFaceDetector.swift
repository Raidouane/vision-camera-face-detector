import Vision
import MLKitFaceDetection
import MLKitVision
import CoreML

@objc(VisionCameraFaceDetector)
public class VisionCameraFaceDetector: NSObject, FrameProcessorPluginBase {


  static var FaceDetectorOption: FaceDetectorOptions = {
    let option = FaceDetectorOptions()
    option.contourMode = .all
    option.performanceMode = .fast
    return option
  }()

  static var faceDetector = FaceDetector.faceDetector(options: FaceDetectorOption)


  private static func processFace(from faces: [Face]?) -> Any  {
    guard let faces = faces else {
        return []
    }
    var faceMap: [Any] = []

    for face in faces {
        var faceCharacteristics: [String: CGFloat] = [:]

        if face.hasHeadEulerAngleX {
           faceCharacteristics["rotX"] = face.headEulerAngleX  // Head is rotated to the uptoward rotX degrees
         }
         if face.hasHeadEulerAngleY {
            faceCharacteristics["rotY"] = face.headEulerAngleY // Head is rotated to the right rotY degrees
         }
         if face.hasHeadEulerAngleZ {
            faceCharacteristics["rotZ"] = face.headEulerAngleZ  // Head is tilted sideways rotZ degrees
         }

        faceMap.append(faceCharacteristics)
    }

    return faceMap
  }

    private static func getImageFromSampleBuffer (buffer:CMSampleBuffer) -> UIImage? {
          if let pixelBuffer = CMSampleBufferGetImageBuffer(buffer) {
              let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
              let uiImage = UIImage(ciImage: ciImage)
              let srcWidth = CGFloat(ciImage.extent.width)
              if (srcWidth <= 720) {
                return uiImage;
              }

              let srcHeight = CGFloat(ciImage.extent.height)
              let dstWidth: CGFloat = 720
              let ratio = dstWidth / srcWidth
              let dstHeight: CGFloat = srcHeight * ratio
              let imageSize = CGSize(width: srcWidth * ratio, height: srcHeight * ratio)
              UIGraphicsBeginImageContextWithOptions(imageSize, false, 1.0)
              uiImage.draw(in: CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight))
              let newImage = UIGraphicsGetImageFromCurrentImageContext()
              UIGraphicsEndImageContext()
              return newImage
          }

          return nil
      }


  @objc
  public static func callback(_ frame: Frame!, withArgs _: [Any]!) -> Any! {
    let imageScaled = getImageFromSampleBuffer(buffer: frame.buffer)
    guard imageScaled != nil else {
        NSLog("NO IMAGE SCALED")
        return [];
    }
    let image = VisionImage.init(image: imageScaled!)
    image.orientation = frame.orientation
    var faces:  [Face]

    do {
        faces =  try faceDetector.results(in: image)
           if (!faces.isEmpty){
               let processedFaces = processFace(from: faces)
               print("processedFaces", processedFaces)
               return processedFaces
           }
       } catch _ {
           return []
       }

     return []
  }
}
