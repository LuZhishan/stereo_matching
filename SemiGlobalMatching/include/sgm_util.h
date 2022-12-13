#include <iostream>

namespace sgm_util
{
	//������������ census���߼�
	// census�任

	/**
	 * \brief census�任
	 * \param source	���룬Ӱ������
	 * \param census	�����censusֵ����
	 * \param width		���룬Ӱ���
	 * \param height	���룬Ӱ���
	 */
	void census_transform_5x5(const uint8_t* source, uint32_t* census, const int32_t& width, const int32_t& height);
	void census_transform_9x7(const uint8_t* source, uint64_t* census, const int32_t& width, const int32_t& height);
	// Hamming����
	uint8_t Hamming32(const uint32_t& x, const uint32_t& y);
	uint8_t Hamming64(const uint64_t& x, const uint64_t& y);

	/**
	 * \brief ����·���ۺ� �� ��
	 * \param img_data			���룬Ӱ������
	 * \param width				���룬Ӱ���
	 * \param height			���룬Ӱ���
	 * \param min_disparity		���룬��С�Ӳ�
	 * \param max_disparity		���룬����Ӳ�
	 * \param p1				���룬�ͷ���P1
	 * \param p2_init			���룬�ͷ���P2_Init
	 * \param cost_init			���룬��ʼ��������
	 * \param cost_aggr			�����·���ۺϴ�������
	 * \param is_forward		���룬�Ƿ�Ϊ������������Ϊ�����ң�������Ϊ���ҵ���
	 */
	void CostAggregateLeftRight(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1,const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	/**
	 * \brief ����·���ۺ� �� ��
	 * \param img_data			���룬Ӱ������
	 * \param width				���룬Ӱ���
	 * \param height			���룬Ӱ���
	 * \param min_disparity		���룬��С�Ӳ�
	 * \param max_disparity		���룬����Ӳ�
	 * \param p1				���룬�ͷ���P1
	 * \param p2_init			���룬�ͷ���P2_Init
	 * \param cost_init			���룬��ʼ��������
	 * \param cost_aggr			�����·���ۺϴ�������
	 * \param is_forward		���룬�Ƿ�Ϊ������������Ϊ���ϵ��£�������Ϊ���µ��ϣ�
	 */
	void CostAggregateUpDown(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	/**
	 * \brief �Խ���1·���ۺϣ�����<->���£��K �I
	 * \param img_data			���룬Ӱ������
	 * \param width				���룬Ӱ���
	 * \param height			���룬Ӱ���
	 * \param min_disparity		���룬��С�Ӳ�
	 * \param max_disparity		���룬����Ӳ�
	 * \param p1				���룬�ͷ���P1
	 * \param p2_init			���룬�ͷ���P2_Init
	 * \param cost_init			���룬��ʼ��������
	 * \param cost_aggr			�����·���ۺϴ�������
	 * \param is_forward		���룬�Ƿ�Ϊ������������Ϊ�����ϵ����£�������Ϊ�����µ����ϣ�
	 */
	void CostAggregateDagonal_1(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	/**
	 * \brief �Խ���2·���ۺϣ�����<->���£��L �J
	 * \param img_data			���룬Ӱ������
	 * \param width				���룬Ӱ���
	 * \param height			���룬Ӱ���
	 * \param min_disparity		���룬��С�Ӳ�
	 * \param max_disparity		���룬����Ӳ�
	 * \param p1				���룬�ͷ���P1
	 * \param p2_init			���룬�ͷ���P2_Init
	 * \param cost_init			���룬��ʼ��������
	 * \param cost_aggr			�����·���ۺϴ�������
	 * \param is_forward		���룬�Ƿ�Ϊ������������Ϊ���ϵ��£�������Ϊ���µ��ϣ�
	 */
	void CostAggregateDagonal_2(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	
	/**
	 * \brief ��ֵ�˲�
	 * \param in				���룬Դ���� 
	 * \param out				�����Ŀ������
	 * \param width				���룬����
	 * \param height			���룬�߶�
	 * \param wnd_size			���룬���ڿ���
	 */
	void MedianFilter(const float* in, float* out, const int32_t& width, const int32_t& height, const int32_t wnd_size);


	/**
	 * \brief �޳�С��ͨ��
	 * \param disparity_map		���룬�Ӳ�ͼ 
	 * \param width				���룬����
	 * \param height			���룬�߶�
	 * \param diff_insame		���룬ͬһ��ͨ���ڵľֲ����ز���
	 * \param min_speckle_aera	���룬��С��ͨ�����
	 * \param invalid_val		���룬��Чֵ
	 */
	void RemoveSpeckles(float* disparity_map, const int32_t& width, const int32_t& height, const int32_t& diff_insame,const uint32_t& min_speckle_aera, const float& invalid_val);
}