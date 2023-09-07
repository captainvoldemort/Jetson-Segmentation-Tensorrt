 def extract_masks_for_classes(self, raw_image_generator, target_classes):
        """
        Extract binary masks for specific classes.

        Args:
            raw_image_generator: A generator that yields input images.
            target_classes: A list of target class indices to extract masks for.

        Returns:
            List of dictionaries, where each dictionary contains the following:
            - 'class_id': The class ID of the extracted mask.
            - 'mask': The binary mask for the specified class.
            - 'image': The original input image.
        """
        results = []

        for image_raw in raw_image_generator:
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)

            # Copy input image to host buffer
            np.copyto(self.host_inputs[0], input_image.ravel())

            # Transfer input data to the GPU
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

            # Run inference
            self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)

            # Transfer predictions back from the GPU
            cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)

            # Synchronize the stream
            self.stream.synchronize()

            # Extract masks for specific classes
            result_boxes, _, _, result_proto_coef = self.post_process(
                self.host_outputs[0], origin_h, origin_w
            )
            masks = self.process_mask(self.host_outputs[1], result_proto_coef, result_boxes, origin_h, origin_w)

            # Create dictionaries for each extracted mask
            for class_id in target_classes:
                results.append({
                    'class_id': class_id,
                    'mask': masks[class_id],
                    'image': image_raw  # You can choose to include the original image
                })

        return results

